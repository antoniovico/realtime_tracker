#define PCL_NO_PRECOMPILE

#include <cluster_track_cpp/ekf.hpp>

#include "rclcpp_components/register_node_macro.hpp"

using namespace std::placeholders;

namespace cluster_track_cpp
{
    EKF::EKF(const rclcpp::NodeOptions &options)
        : Node("simple_ekf", options)
    {
        init();
    }

    void EKF::init()
    {
        // put parameters from launch file into node
        cloud_in_topic = this->declare_parameter("input_clustered_pc", "");
        cloud_out_topic = this->declare_parameter("cloud_output_topic", "");
        centroids_out_topic = this->declare_parameter("centroids_output_topic", "");
        wo_measurement_age = this->declare_parameter("input_wo_measurement_age", 10);
        association_distance = this->declare_parameter("input_association_distance", 2.0);
        prediction_error_factor = this->declare_parameter("input_prediction_error_factor", 1.5);    // ekf process noise = (input_prediction_error_factor * I)
        measurement_error_factor = this->declare_parameter("input_measurement_error_factor", 1e-1); // ekf measurement noise = (input_measurement_error_factor * I)
        dont_trust_factor = this->declare_parameter("input_dont_trust_factor", 3);                  // if variance of x_pred or y_pred > dont_trust_factor -> doesnt count as measurement
        ekf_coordinate_system = this->declare_parameter("input_ekf_coordinate_system", "xy");       // input "xy" for cartesian and "rphi" for polar

        // subscription on pointcloud topic
        clustered_cloud_subscriber = this->create_subscription<sensor_msgs::msg::PointCloud2>(
            cloud_in_topic, 10, std::bind(&EKF::ekfTracking, this, _1));

        marker_publisher = this->create_publisher<visualization_msgs::msg::MarkerArray>("/ekf_marker", 10);

        // matrix initialization
        // process noise
        Q << 1, 0, 0, 0, 0,
            0, 1, 0, 0, 0,
            0, 0, 1, 0, 0,
            0, 0, 0, 1, 0,
            0, 0, 0, 0, 1;
        Q = Q * prediction_error_factor;

        // measurement noise
        // covariance is not 0 in xy coordinate system
        if (ekf_coordinate_system == "xy")
        {
            R << 1, -0.05,
                -0.05, 1;
        }
        // covariance is 0 for rphi coordinate system
        else if (ekf_coordinate_system == "rphi")
        {
            R << 1, 0,
                0, 1;
        }
        R = R * measurement_error_factor;

        F = Eigen::MatrixXd::Identity(5, 5);
        H = Eigen::MatrixXd::Zero(2, 5);

        // publisher for merged point cloud and for centroids
        pub_cloud_out = this->create_publisher<sensor_msgs::msg::PointCloud2>(cloud_out_topic, 10);
        pub_centroids = this->create_publisher<sensor_msgs::msg::PointCloud2>(centroids_out_topic, 10);
        pub_tracklist = this->create_publisher<track_msgs::msg::TrackList>("/custom_tracklist", 10);
    }

    void EKF::ekfTracking(const sensor_msgs::msg::PointCloud2::SharedPtr clustered_cloud_msg)
    {
        auto timer_start = std::chrono::high_resolution_clock::now();

        pcl::PointCloud<PointTCrit>::Ptr pcl_cloud(new pcl::PointCloud<PointTCrit>);
        pcl::fromROSMsg(*clustered_cloud_msg, *pcl_cloud);

        pcl::PointCloud<pcl::PointXYZ>::Ptr pcl_centroids(new pcl::PointCloud<pcl::PointXYZ>());
        centroid_to_clusterid.clear();

        // Map Centroid_ID: [Point1, Point2, ...] (Point1, Point2, ... belong to the same cluster)
        for (unsigned int i = 0; i < pcl_cloud->points.size(); ++i)
        {
            if (pcl_cloud->points[i].cluster_ID > -1)
            {
                centroid_to_clusterid[pcl_cloud->points[i].cluster_ID].push_back(i);
            }
        }

        // Calculate centroid position for each cluster
        pcl::PointXYZ cntrd_pt;
        for (const auto &centroid : centroid_to_clusterid)
        {
            x_centroid = 0.0;
            y_centroid = 0.0;
            for (const auto &pt_idx : centroid.second)
            {
                x_centroid += pcl_cloud->points[pt_idx].x;
                y_centroid += pcl_cloud->points[pt_idx].y;
            }
            cntrd_pt.x = x_centroid / centroid.second.size();
            cntrd_pt.y = y_centroid / centroid.second.size();
            cntrd_pt.z = 0.0; // TODO: what is this for
            pcl_centroids->push_back(cntrd_pt);
        }

        // Calculate dt for EKF algorithm
        cur_timestamp = clustered_cloud_msg->header.stamp.sec * 1e6 + clustered_cloud_msg->header.stamp.nanosec * 1e-3;
        RCLCPP_INFO(this->get_logger(), "cur_timestamp: %lu, last_timestamp: %lu, dt: %.30f", cur_timestamp, last_timestamp, dt);
        if (first_run)
        {
            first_run = false;
            last_timestamp = cur_timestamp;
            return;
        }
        dt = (cur_timestamp - last_timestamp) * 1e-6;
        Eigen::MatrixXd result;

        // double merge_distance = 1.0;  // Define a reasonable threshold for merging

        // for (long int i = 0; i < cur_tracks.size(); i++) {
        //     for (long int j = i + 1; j < cur_tracks.size(); j++) {
        //         Eigen::Vector2d pos_i(cur_tracks[i].x_pred[0], cur_tracks[i].x_pred[1]);
        //         Eigen::Vector2d pos_j(cur_tracks[j].x_pred[0], cur_tracks[j].x_pred[1]);

        //         double distance = (pos_i - pos_j).norm();  // Euclidean distance

        //         if (distance < merge_distance) {  // Check if tracks are close
        //             // Merge logic: weighted average based on covariance
        //             Eigen::MatrixXd P1 = cur_tracks[i].P_pred;
        //             Eigen::MatrixXd P2 = cur_tracks[j].P_pred;
        //             Eigen::MatrixXd P_inv_sum = (P1.inverse() + P2.inverse()).inverse();

        //             Eigen::VectorXd x_merged = P_inv_sum * (P1.inverse() * cur_tracks[i].x_pred +
        //                                                     P2.inverse() * cur_tracks[j].x_pred);

        //             cur_tracks[i].x_pred = x_merged;
        //             cur_tracks[i].P_pred = P_inv_sum;  // Update covariance

        //             // Remove track j after merging
        //             cur_tracks.erase(cur_tracks.begin() + j);
        //             j--;  // Adjust index after deletion
        //         }
        //     }
        // }

        // Tracks to be "tracked" for each centroid
        Track new_track;
        Track *min_track = nullptr;
        for (const auto &centroid : pcl_centroids->points)
        {
            // Association between measured centroids at time t and predicted centroids at time t (predicted at t-1)
            // centroid = measured centroid's position at time t
            // cur_tracks.x_pred = predicted centroid's position at time t
            // Calculate distance = min(centroid - cur_tracks[i].x_pred for all i)
            // Set upper limit for association
            // case 1: associate -> found new centroid without track id -> create track and predict
            // case 2: associate -> add to centroid to track -> update and predict
            double min_distance = std::numeric_limits<double>::max();
            // int min_id = -1; // track id of the track with minimal distance to centroid
            min_track = nullptr;

            double distance;
            for (auto &track : cur_tracks)
            {

                // z_pred = (x, y) position of object currently investigated
                z_pred << track.x_pred[0], track.x_pred[1];
                // innovation covariance matrix
                // S = track.P_pred.block<2, 2>(0, 0);
                // S = Eigen::MatrixXd::Identity(2, 2); // debug: use euclidean distance
                // S += Eigen::MatrixXd::Identity(2, 2) * 1e-3; // Regularize to avoid singularity
                // innovation vector = measurement - prediction
                y << centroid.x - z_pred[0], centroid.y - z_pred[1];

                // determine distance
                distance = y.norm();
                if (distance < min_distance)
                {
                    min_distance = distance;
                    min_track = &track;
                }
            }

            // Case 1: New centroid to track
            double min_distance_upper_limit = association_distance;
            if (min_distance > min_distance_upper_limit)
            {
                // if (min_distance != std::numeric_limits<double>::max()){
                // RCLCPP_INFO(this->get_logger(), "distance: %f", min_distance);}
                new_track.x_pred << centroid.x, centroid.y, 0.0, 0.0, 0.0;
                new_track.P_pred = Eigen::MatrixXd::Identity(5, 5) * 1e-1;
                int previous_track_id = -1;
                bool gap_found = false;
                std::list<int> track_id_list;
                for (const auto &track : cur_tracks)
                {
                    track_id_list.push_back(track.track_id);
                }
                track_id_list.sort();

                for (const auto &track_id : track_id_list)
                {
                    if (track_id - previous_track_id > 1)
                    {
                        // Found a gap
                        new_track.track_id = previous_track_id + 1;
                        gap_found = true;
                        break;
                    }
                    previous_track_id = track_id;
                }

                // If no gap is found, assign the next sequential ID
                if (!gap_found)
                {
                    new_track.track_id = previous_track_id + 1;
                }

                // new_track.track_id = cur_tracks.size();
                new_track.track_age = -1;
                new_track.track_wo_measurement = -1; // will be added again to zero in (CLEAN UP OBJECTS)
                cur_tracks.push_back(new_track);
                RCLCPP_INFO(this->get_logger(), "distance: %f", min_distance);
            }

            // Case 2: Centroid exist in the past already
            else
            {
                if (min_track != nullptr)
                    RCLCPP_INFO(this->get_logger(), "distance: %f", min_distance);
                update(*min_track, centroid);         // update for THIS time step // cur_tracks is the preidction for THIS
                                                      // timestep (calculated at the timestep before)
                min_track->track_wo_measurement = -1; // will be added again to zero in (CLEAN UP OBJECTS)
            }
        }

        // if the "trustworthiness" of a tracked object is not good, it counts as note measured (if happens too often, it will be deleted)
        // for (auto &track: cur_tracks){
        //     if (track.P_pred(0,0) > dont_trust_factor || track.P_pred(1,1) > dont_trust_factor){
        //         // track.track_wo_measurement++;
        //         track.track_wo_measurement = wo_measurement_age + 1;
        //     }
        // }

        // for (auto &track: cur_tracks){
        //     if (2 * track.track_wo_measurement > track.track_age){
        //         // track.track_wo_measurement++;
        //         track.track_wo_measurement = wo_measurement_age + 1;
        //     }
        // }
        deleteTracks(cur_tracks, wo_measurement_age);
        publishMarkers();
        publishTracks(clustered_cloud_msg->header.stamp);
        // RCLCPP_INFO(this->get_logger(), "published");

        for (auto &track : cur_tracks)
        {
            if (track.track_id != -1)
            {
                predict(track, dt); // prediction for the NEXT time step
            }
        }

        last_timestamp = cur_timestamp;

        auto timer_stop = std::chrono::high_resolution_clock::now();
        auto tracking_duration = std::chrono::duration_cast<std::chrono::microseconds>(timer_stop - timer_start);
        // RCLCPP_INFO(this->get_logger(), "Tracking duration: %ld microseconds", tracking_duration.count());

        // for (const auto &track : cur_tracks)
        // {
        //     // Log is not sorted!
        //     RCLCPP_INFO(this->get_logger(), "Tracked Objects: %lu, TrackSpeed: %f, TrackID: %d, TrackAge: %d, TrackWoMeasurementAge: %d",
        //                                         cur_tracks.size(), track.x_pred[2], track.track_id, track.track_age, track.track_wo_measurement);
        // }
    }

    void EKF::deleteTracks(std::vector<Track> &cur_tracks, const int wo_measurement_age)
    {
        for (auto it = cur_tracks.begin(); it != cur_tracks.end();)
        {

            it->track_age++;
            it->track_wo_measurement++;
            if (it->track_wo_measurement >= wo_measurement_age)
            {
                it = cur_tracks.erase(it); // iterator jumps to the next element after deletion
            }
            else
            {
                ++it;
            }
        }
    }

    void EKF::predict(Track &track, double dt)
    {
        // current track state attributes
        // state vector [x, y, v, yaw, yawrate]
        x_pos = track.x_pred[0];
        y_pos = track.x_pred[1];
        v = track.x_pred[2];
        // RCLCPP_INFO(this->get_logger(), "speed:%f", v);
        track.x_pred[3] = std::atan2(std::sin(track.x_pred[3]), std::cos(track.x_pred[3]));
        yaw = track.x_pred[3];
        yawd = track.x_pred[4];
        Eigen::Matrix<double, 5, 1> delta_x_pred;

        // Case: yaw rate = 0
        if (abs(yawd) < 1e-6)
        {
            F(0, 2) = dt * std::cos(yaw);
            // RCLCPP_INFO(this->get_logger(), "first: dt: %f, yaw: %f, cos(yaw): %f", dt, yaw, std::cos(yaw));
            F(0, 3) = -dt * v * std::sin(yaw);
            F(0, 4) = -pow(dt, 2) * v * std::sin(yaw);
            F(1, 2) = dt * std::sin(yaw);
            F(1, 3) = dt * v * std::cos(yaw);
            F(1, 4) = pow(dt, 2) * v * std::cos(yaw);
            // F(3,4) = dt;

            delta_x_pred << dt * v * std::cos(yaw),
                dt * v * std::sin(yaw),
                0,
                dt * yawd,
                0;
        }
        else
        {
            // Case: yaw rate non zero
            F(0, 2) = (-std::sin(yaw) + std::sin(yaw + yawd * dt)) / yawd;
            // RCLCPP_INFO(this->get_logger(), "second: dt: %f, yaw: %f, yawd: %f", dt, yaw, yawd);
            F(0, 3) = (-v * std::cos(yaw) + v * std::cos(yaw + yawd * dt)) / yawd;
            F(0, 4) = (v * std::sin(yaw) - v * std::sin(yaw + yawd * dt)) / pow(yawd, 2) + v * std::cos(yaw + yawd * dt) * dt / yawd;
            F(1, 2) = (std::cos(yaw) - std::cos(yaw + yawd * dt)) / yawd;
            F(1, 3) = (-v * std::sin(yaw) + v * std::sin(yaw + yawd * dt)) / yawd;
            F(1, 4) = (-v * std::cos(yaw) + v * std::cos(yaw + yawd * dt)) / pow(yawd, 2) + v * std::sin(yaw + yawd * dt) * dt / yawd;
            F(3, 4) = dt;

            // prediction step
            delta_x_pred << -v * std::sin(yaw) / (yawd) + v * std::sin(yaw + yawd * dt) / yawd,
                v * std::cos(yaw) / yawd - v * std::cos(yaw + yawd * dt) / yawd,
                0,
                yawd * dt,
                0;
        }

        // RCLCPP_INFO(this->get_logger(), "Matrix F: ");
        //     for (long int i = 0; i < track.P_pred.rows(); ++i) {
        //         for (long int j = 0; j < track.P_pred.cols(); ++j) {
        //             RCLCPP_INFO(this->get_logger(), "F[%zu][%zu] = %f", i, j, F(i, j));
        //         }
        //     }

        track.x_pred += delta_x_pred;
        // RCLCPP_INFO(this->get_logger(), "Matrix P before: ");
        //     for (long int i = 0; i < track.P_pred.rows(); ++i) {
        //         for (long int j = 0; j < track.P_pred.cols(); ++j) {
        //             RCLCPP_INFO(this->get_logger(), "P[%zu][%zu] = %f", i, j, track.P_pred(i, j));
        //         }
        //     }
        track.P_pred = F * track.P_pred * F.transpose() + Q;
        // RCLCPP_INFO(this->get_logger(), "Matrix P: ");
        //     for (long int i = 0; i < track.P_pred.rows(); ++i) {
        //         for (long int j = 0; j < track.P_pred.cols(); ++j) {
        //             RCLCPP_INFO(this->get_logger(), "Ppred[%zu][%zu] = %f", i, j, track.P_pred(i, j));
        //         }
        //     }
    }

    void EKF::update(Track &track, const pcl::PointXYZ &measurement)
    {
        if (ekf_coordinate_system == "rphi") // observation in polar coordinates
        {
            Eigen::Vector2f z;
            z << measurement.x, measurement.y;

            double epsilon = 1e-6; // variable to avoid numerical instability

            // Compute polar coordinates
            double r = sqrt(pow(z(0), 2) + pow(z(1), 2) + epsilon);
            // r = r  100; // to avoid numerical instability when inversing because matrix is bad conditioned
            double theta = atan2(z(1), z(0));
            m << r, theta;
            float dr_dx = z(0) / r;
            float dr_dy = z(1) / r;
            float dtheta_dx = -z(1) / pow(r, 2);
            float dtheta_dy = z(0) / pow(r, 2);

            H(0, 0) = dr_dx;
            H(0, 1) = dr_dy;
            H(1, 0) = dtheta_dx;
            H(1, 1) = dtheta_dy;
            // Innovation covariance
            S = H * track.P_pred * H.transpose() + R;
            RCLCPP_INFO(this->get_logger(), "Matrix S: ");
            for (long int i = 0; i < S.rows(); ++i)
            {
                for (long int j = 0; j < S.cols(); ++j)
                {
                    RCLCPP_INFO(this->get_logger(), "S[%zu][%zu] = %f", i, j, S(i, j));
                }
            }

            // // Log the matrix H
            // RCLCPP_INFO(this->get_logger(), "Matrix H: ");
            // for (long int i = 0; i < H.rows(); ++i) {
            //     for (long int j = 0; j < H.cols(); ++j) {
            //         RCLCPP_INFO(this->get_logger(), "H[%zu][%zu] = %f", i, j, H(i, j));
            //     }
            // }

            // Log the matrix P (track.P_pred)
            RCLCPP_INFO(this->get_logger(), "Matrix P: ");
            for (long int i = 0; i < track.P_pred.rows(); ++i)
            {
                for (long int j = 0; j < track.P_pred.cols(); ++j)
                {
                    RCLCPP_INFO(this->get_logger(), "P[%zu][%zu] = %f", i, j, track.P_pred(i, j));
                }
            }
            // S += Eigen::MatrixXd::Identity(2, 2);
            // Optimal Kalman Gain
            K = track.P_pred * H.transpose() * S.transpose();
            // Log the matrix K
            RCLCPP_INFO(this->get_logger(), "Matrix K: ");
            for (long int i = 0; i < K.rows(); ++i)
            {
                for (long int j = 0; j < K.cols(); ++j)
                {
                    RCLCPP_INFO(this->get_logger(), "K[%zu][%zu] = %f", i, j, K(i, j));
                }
            }
            // m(0) *= 100; // after inverse, change the scale back; for correct calculation
            Eigen::Matrix<double, 2, 1> h;
            h << sqrt(pow(track.x_pred[0], 2) + pow(track.x_pred[1], 2)),
                atan2(track.x_pred[1], track.x_pred[0]);
            // Update state vector
            track.x_pred += K * (m - h);
            // Update covariance estimate
            track.P_pred = (Eigen::MatrixXd::Identity(5, 5) - K * H) * track.P_pred;
            // RCLCPP_INFO(this->get_logger(), "Matrix P: ");
            // for (long int i = 0; i < track.P_pred.rows(); ++i) {
            //     for (long int j = 0; j < track.P_pred.cols(); ++j) {
            //         RCLCPP_INFO(this->get_logger(), "P[%zu][%zu] = %f", i, j, track.P_pred(i, j));
            //     }
            // }
        }

        else if (ekf_coordinate_system == "xy") // observation in xy coordinates
        {
            H(0, 0) = 1;
            H(1, 1) = 1;

            S = H * track.P_pred * H.transpose() + R;

            // RCLCPP_INFO(this->get_logger(), "Matrix P: ");
            // for (long int i = 0; i < track.P_pred.rows(); ++i) {
            //     for (long int j = 0; j < track.P_pred.cols(); ++j) {
            //         RCLCPP_INFO(this->get_logger(), "P[%zu][%zu] = %f", i, j, track.P_pred(i, j));
            //     }
            // }
            // RCLCPP_INFO(this->get_logger(), "Matrix S: ");
            // for (long int i = 0; i < S.rows(); ++i) {
            //     for (long int j = 0; j < S.cols(); ++j) {
            //         RCLCPP_INFO(this->get_logger(), "S[%zu][%zu] = %f", i, j, S(i, j));
            //     }
            // }

            // Sinverse = S.inverse();
            // RCLCPP_INFO(this->get_logger(), "Matrix S: ");
            // for (long int i = 0; i < S.rows(); ++i) {
            //     for (long int j = 0; j < S.cols(); ++j) {
            //         RCLCPP_INFO(this->get_logger(), "S[%zu][%zu] = %f", i, j, Sinverse(i, j));
            //     }
            // }

            K = track.P_pred * H.transpose() * S.inverse();

            // RCLCPP_INFO(this->get_logger(), "Matrix K: ");
            // for (long int i = 0; i < S.rows(); ++i) {
            //     for (long int j = 0; j < S.cols(); ++j) {
            //         RCLCPP_INFO(this->get_logger(), "K[%zu][%zu] = %f", i, j, K(i, j));
            //     }
            // }

            Eigen::Matrix<double, 2, 1> h;
            h << track.x_pred[0],
                track.x_pred[1];
            m << measurement.x,
                measurement.y;
            track.x_pred += K * (m - h);
            track.P_pred = (Eigen::MatrixXd::Identity(5, 5) - K * H) * track.P_pred;
            // track.P_pred += Eigen::MatrixXd::Ones(5, 5) * 1e-10;
            // track.P_pred = 0.5 * (track.P_pred + track.P_pred.transpose());
            // RCLCPP_INFO(this->get_logger(), "Matrix P: ");
            // for (long int i = 0; i < track.P_pred.rows(); ++i) {
            //     for (long int j = 0; j < track.P_pred.cols(); ++j) {
            //         RCLCPP_INFO(this->get_logger(), "Pupdt[%zu][%zu] = %f", i, j, track.P_pred(i, j));
            //     }
            // }
        }
    }

    void EKF::publishMarkers()
    {
        markers.markers.clear();
        for (const auto &track : cur_tracks)
        {
            if (track.track_age >= -1)
            {
                marker.header.frame_id = "base_link";
                marker.header.stamp = now();
                marker.ns = "tracked_objects" + std::to_string(track.track_id);
                marker.id = track.track_id;
                marker.type = visualization_msgs::msg::Marker::ARROW;
                marker.action = visualization_msgs::msg::Marker::ADD;
                marker.pose.position.x = track.x_pred[0];
                marker.pose.position.y = track.x_pred[1];
                marker.pose.position.z = 0;
                marker.pose.orientation.x = 0.0;
                marker.pose.orientation.y = 0.0;
                marker.pose.orientation.z = std::sin(track.x_pred[3] / 2);
                marker.pose.orientation.w = std::cos(track.x_pred[3] / 2);
                marker.scale.x = 1.5; // track.x_pred[2] / std::abs(track.x_pred[2]) * std::max(std::abs(track.x_pred[2]) / 3.0, 0.5);
                marker.scale.y = 0.2;
                marker.scale.z = 0.4;
                marker.color.a = 1.0;
                marker.color.r = 0.0;
                marker.color.g = 1.0;
                marker.color.b = 0.0;
                marker.lifetime = rclcpp::Duration(0, 500000000);
                markers.markers.push_back(marker);
            }
        }
        marker_publisher->publish(markers);
    }

    void EKF::publishTracks(builtin_interfaces::msg::Time cur_stamp)
    {
        tracklist_obj.header.frame_id = "base_link";
        tracklist_obj.header.stamp = cur_stamp;
        tracklist_obj.no_tracks = cur_tracks.size();
        tracklist_obj.tracks.clear();

        for (const auto &track : cur_tracks)
        {
            track_obj.pose.position.x = track.x_pred[0];
            track_obj.pose.position.y = track.x_pred[1];
            track_obj.pose.position.z = 0.0;
            track_obj.pose.orientation.x = 0.0;
            track_obj.pose.orientation.y = 0.0;
            track_obj.pose.orientation.z = std::sin(track.x_pred[3] / 2);
            track_obj.pose.orientation.w = std::cos(track.x_pred[3] / 2);
            track_obj.v_x = track.x_pred[2];
            track_obj.yawrate = track.x_pred[4];
            track_obj.ttc = 0.0;
            track_obj.id = track.track_id;
            track_obj.age = track.track_age;
            track_obj.cycles_wo_measurement = track.track_wo_measurement;
            tracklist_obj.tracks.push_back(track_obj);
            // RCLCPP_INFO(this->get_logger(), "Tracked Objects: %lu, TrackID: %d, TrackAge: %d, TrackWoMeasurementAge: %d, yaw: %f",
            //             cur_tracks.size(), track.track_id, track.track_age, track.track_wo_measurement, track.x_pred[3]);
        }
        pub_tracklist->publish(tracklist_obj);
    }

    Eigen::MatrixXd EKF::createAssociation(const std::vector<Track> cur_tracks,
                                           const pcl::PointCloud<pcl::PointXYZ>::Ptr pcl_centroids, float non_assignment_cost)
    {
        // std::map<int, int> // TODO

        // create cost matrix (M x N : Tracks x Detection)
        num_tracks = cur_tracks.size();
        num_detections = pcl_centroids->size();
        num_dummy_tracks = num_detections; // num dummy tracks and detections for the NO associations
        num_dummy_detections = num_tracks; // each track(detection) has one non-assignment cost
        cost_matrix.resize(num_tracks + num_dummy_tracks, num_detections + num_dummy_detections);
        cost_matrix.setZero();

        // set value to the cost_matrix
        // cost is the euclidean distance between the predicted position of tracks and measured position of centroids
        for (long int i = 0; i < num_tracks; i++)
        {
            pos_pred << cur_tracks[i].x_pred[0],
                cur_tracks[i].x_pred[1];
            for (long int j = 0; j < num_detections; j++)
            {
                pos_meas << pcl_centroids->points[j].x,
                    pcl_centroids->points[j].y;
                cost = (pos_pred - pos_meas).norm();
                cost_matrix(i, j) = cost;
            }
        }

        // set non association block
        // // prepare non association block
        cost_matrix.block(0, num_detections, num_tracks, num_dummy_detections).setConstant(std::numeric_limits<double>::infinity());
        cost_matrix.block(num_tracks, 0, num_dummy_tracks, num_detections).setConstant(std::numeric_limits<double>::infinity());
        // // set cost for non association
        cost_matrix.block(0, num_detections, num_tracks, num_dummy_detections).diagonal().setConstant(non_assignment_cost);
        cost_matrix.block(num_tracks, 0, num_dummy_tracks, num_detections).diagonal().setConstant(non_assignment_cost);

        // hungarian algorithm
        // // minimize row
        double row_min, col_min;
        for (long int i = 0; i < cost_matrix.rows(); i++)
        {
            row_min = std::numeric_limits<double>::infinity();
            for (long int j = 0; j < cost_matrix.cols(); j++)
            {
                if (cost_matrix(i, j) < row_min)
                {
                    row_min = cost_matrix(i, j);
                }
            }
            cost_matrix.row(i).array() -= row_min;
        }

        // // minimize column
        for (long int i = 0; i < cost_matrix.cols(); i++)
        {
            col_min = std::numeric_limits<double>::infinity();
            for (long int j = 0; j < cost_matrix.rows(); j++)
            {
                if (cost_matrix(j, i) < col_min)
                {
                    col_min = cost_matrix(j, i);
                }
            }
            cost_matrix.col(i).array() -= col_min;
        }

        // // search for the most cost efficient route
        // // // register the i and j index of elements with value "0"
        for (long int i = 0; i < cost_matrix.rows(); i++)
        {
            for (long int j = 0; j < cost_matrix.cols(); j++)
            {
            }
        }

        // // return centroid_to_trackid;
        return cost_matrix;
    }

} // namespace cluster_track_cpp

// Register the component with class_loader.
// This acts as a sort of entry point, allowing the component to be discoverable
// when its library is being loaded into a running process.
RCLCPP_COMPONENTS_REGISTER_NODE(cluster_track_cpp::EKF)