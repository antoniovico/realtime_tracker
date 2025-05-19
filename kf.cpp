#define PCL_NO_PRECOMPILE

#include <cluster_track_cpp/kf.hpp>

#include "rclcpp_components/register_node_macro.hpp"

using namespace std::placeholders;

namespace cluster_track_cpp
{
    KF::KF(const rclcpp::NodeOptions &options)
        : Node("kf", options)
    {
        init();
    }

    void KF::init()
    {
        // put parameters from launch file into node
        cloud_in_topic = this->declare_parameter("input_clustered_pc", "");
        cloud_out_topic = this->declare_parameter("cloud_output_topic", "");
        centroids_out_topic = this->declare_parameter("centroids_output_topic", "");
        wo_measurement_age = this->declare_parameter("input_wo_measurement_age", 10);
        association_distance = this->declare_parameter("input_association_distance", 2.0);
        prediction_error_factor = this->declare_parameter("input_prediction_error_factor", 1.5);    // KF process noise = (input_prediction_error_factor * I)
        measurement_error_factor = this->declare_parameter("input_measurement_error_factor", 1e-1); // KF measurement noise = (input_measurement_error_factor * I)
        ttc_upper_limit = this->declare_parameter("input_ttc_upper_limit", 3.0);
        crit_limit = this->declare_parameter("input_crit_limit", 0.5);

        // subscription on pointcloud topic
        clustered_cloud_subscriber = this->create_subscription<sensor_msgs::msg::PointCloud2>(
            cloud_in_topic, 10, std::bind(&KF::kfTracking, this, _1));
        // subscription to EML01 topic on FlexRay
        eml01_subscriber = this->create_subscription<flexdev_interfaces::msg::EML01>(
            "FR/ZFAS/EML01", 10, std::bind(&KF::eml01_callback, this, _1));

        marker_publisher = this->create_publisher<visualization_msgs::msg::MarkerArray>("/kf_marker", 10);

        // matrix initialization
        // process noise
        Q.setZero(); // set value in KF::kfTracking(), bcs of dependency on dt

        // measurement noise
        R << 1, -0.05,
            -0.05, 1;
        R = R * measurement_error_factor;

        v_ego = 0.0;

        F = Eigen::MatrixXd::Identity(4, 4);
        H = Eigen::MatrixXd::Zero(2, 4);

        // publisher for merged point cloud and for centroids
        pub_cloud_out = this->create_publisher<sensor_msgs::msg::PointCloud2>(cloud_out_topic, 10);
        pub_centroids = this->create_publisher<sensor_msgs::msg::PointCloud2>(centroids_out_topic, 10);
        pub_tracklist = this->create_publisher<track_msgs::msg::TrackList>("/custom_tracklist", 10);
    }

    void KF::eml01_callback(const flexdev_interfaces::msg::EML01::SharedPtr eml01_msg)
    {
        v_ego = eml01_msg->eml_geschwx.phy;
    }

    void KF::kfTracking(const sensor_msgs::msg::PointCloud2::SharedPtr clustered_cloud_msg)
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
            z_centroid = 0.0; // z component: criticality value
            for (const auto &pt_idx : centroid.second)
            {
                x_centroid += pcl_cloud->points[pt_idx].x;
                y_centroid += pcl_cloud->points[pt_idx].y;
                z_centroid += pcl_cloud->points[pt_idx].z;
            }
            cntrd_pt.x = x_centroid / centroid.second.size();
            cntrd_pt.y = y_centroid / centroid.second.size();
            cntrd_pt.z = z_centroid / centroid.second.size();
            pcl_centroids->push_back(cntrd_pt);
        }

        // Calculate dt for KF algorithm
        cur_timestamp = clustered_cloud_msg->header.stamp.sec * 1e6 + clustered_cloud_msg->header.stamp.nanosec * 1e-3;
        if (first_run)
        {
            first_run = false;
            last_timestamp = cur_timestamp;
            return;
        }
        dt = (cur_timestamp - last_timestamp) * 1e-6;

        // // Set value for process noise Q
        Eigen::Matrix<double, 4, 1> dr;
        double sigma_x, sigma_y;
        sigma_x = 0.16; // std of acceleration in x
        sigma_y = 0.16; // std of acceleration in y

        Q(0, 0) = 0.25 * pow(dt, 4) * pow(sigma_x, 2);
        Q(0, 2) = 0.5 * pow(dt, 3) * pow(sigma_x, 2);
        Q(1, 1) = 0.25 * pow(dt, 4) * pow(sigma_y, 2);
        Q(1, 3) = 0.5 * pow(dt, 3) * pow(sigma_y, 2);
        Q(2, 0) = 0.5 * pow(dt, 3) * pow(sigma_x, 2);
        Q(2, 2) = pow(dt, 2) * pow(sigma_y, 2);
        Q(3, 1) = 0.5 * pow(dt, 3) * pow(sigma_y, 2);
        Q(3, 3) = pow(dt, 2) * pow(sigma_y, 2);

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
            min_track = nullptr;

            double distance;
            for (auto &track : cur_tracks)
            {

                // z_pred = (x, y) position of object currently investigated
                z_pred << track.x_pred[0], track.x_pred[1];
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
                new_track.x_pred << centroid.x, centroid.y, 0.0, 0.0;
                new_track.P_pred = Eigen::MatrixXd::Identity(4, 4) * measurement_error_factor;
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

                new_track.track_age = -1;
                new_track.track_wo_measurement = -1; // will be added again to zero in (CLEAN UP OBJECTS)
                new_track.crit = centroids.z;
                cur_tracks.push_back(new_track);
            }

            // Case 2: Centroid exist in the past already
            else
            {
                if (min_track != nullptr)
                {
                    min_track->crit = centroids.z;
                    update(*min_track, centroid);         // update for THIS time step // cur_tracks is the preidction for THIS
                                                          // timestep (calculated at the timestep before)
                    min_track->track_wo_measurement = -1; // will be added again to zero in (CLEAN UP OBJECTS)
                }
            }
        }

        for (auto &track : cur_tracks)
        {
            if (track.track_age - track.track_wo_measurement <= 2 && track.track_age>=2 && track.crit < crit_limit)
            {
                track.track_wo_measurement = wo_measurement_age + 1; // direct delete
            }
        }

        deleteTracks(cur_tracks, wo_measurement_age);
        calcTTC(cur_tracks, v_ego, ttc_upper_limit);
        publishMarkers();
        publishTracks(clustered_cloud_msg->header.stamp);

        // publish centroids determined in this function
        pcl::toROSMsg(*pcl_centroids, output_centroids);
        output_centroids.header.stamp = clustered_cloud_msg->header.stamp; // Use Header from reference
        output_centroids.header.frame_id = "base_link";
        pub_centroids->publish(output_centroids);

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

        for (const auto &track : cur_tracks)
        {
            // Log is not sorted!
            RCLCPP_INFO(this->get_logger(), "Tracked Objects: %lu, TrackID: %d, TTC: %f, TrackAge: %d, TrackWoMeasurementAge: %d",
                        cur_tracks.size(), track.track_id, track.ttc, track.track_age, track.track_wo_measurement);
        }
    }

    void KF::deleteTracks(std::vector<Track> &cur_tracks, const int wo_measurement_age)
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

    void KF::predict(Track &track, double dt)
    {
        // current track state attributes
        // state vector [x, y, v_x, v_y]
        x_pos = track.x_pred[0];
        y_pos = track.x_pred[1];
        v_x = track.x_pred[2];
        v_y = track.x_pred[3];
        F = Eigen::MatrixXd::Identity(4, 4);
        F(0, 2) = dt;
        F(1, 3) = dt;

        track.x_pred = F * track.x_pred;
        track.x_pred[0] -= dt * v_ego;

        track.P_pred = F * track.P_pred * F.transpose() + Q;
    }

    void KF::update(Track &track, const pcl::PointXYZ &measurement)
    {
        // observation in xy coordinates
        H(0, 0) = 1;
        H(1, 1) = 1;

        S = H * track.P_pred * H.transpose() + R;

        K = track.P_pred * H.transpose() * S.inverse();

        Eigen::Matrix<double, 2, 1> h;
        h << track.x_pred[0],
            track.x_pred[1];
        m << measurement.x,
            measurement.y;
        track.x_pred += K * (m - h);
        track.P_pred = (Eigen::MatrixXd::Identity(4, 4) - K * H) * track.P_pred;
        track.P_pred = 0.5 * (track.P_pred + track.P_pred.transpose()); // attempt to minimalize instabilities
    }
    void KF::publishMarkers()
    {
        markers.markers.clear();
        for (const auto &track : cur_tracks)
        {
            if (track.track_age >= 3 || track.crit > crit_limit)
            {
                v_track = std::sqrt(std::pow(track.x_pred[2], 2) + std::pow(track.x_pred[3], 2));
                marker.header.frame_id = "base_link";
                marker.header.stamp = now();
                marker.ns = "tracked_objects" + std::to_string(track.track_id);
                marker.id = track.track_id;
                if (v_track < 0.1)
                {
                    marker.type = visualization_msgs::msg::Marker::SPHERE;
                    marker.scale.x = 0.5;
                    marker.scale.y = 0.5;
                    marker.scale.z = 0.5;
                }
                else
                {
                    marker.type = visualization_msgs::msg::Marker::ARROW;
                    marker.scale.x = std::sqrt(std::pow(track.x_pred[2], 2) + std::pow(track.x_pred[3], 2)) * 1.5;
                    marker.scale.y = 0.2;
                    marker.scale.z = 0.4;
                }
                marker.action = visualization_msgs::msg::Marker::ADD;
                marker.pose.position.x = track.x_pred[0];
                marker.pose.position.y = track.x_pred[1];
                marker.pose.position.z = 0;
                yawangle_track = std::atan2(track.x_pred[3], track.x_pred[2]);
                marker.pose.orientation.x = 0.0;
                marker.pose.orientation.y = 0.0;
                marker.pose.orientation.z = std::sin(yawangle_track / 2);
                marker.pose.orientation.w = std::cos(yawangle_track / 2);
                marker.color.a = 1.0;
                if (track.track_age > 2)
                {
                    marker.color.r = 0.0;
                    marker.color.g = 1.0;
                    marker.color.b = 0.0;
                }
                else
                {
                    marker.color.r = 1.0;
                    marker.color.g = 0.0;
                    marker.color.b = 0.0;
                }
                marker.lifetime = rclcpp::Duration(0, 500000000);
                markers.markers.push_back(marker);

                // text marker
                marker.ns = "text_tracked_objects" + std::to_string(track.track_id);
                marker.type = visualization_msgs::msg::Marker::TEXT_VIEW_FACING;
                // marker.id += 1000;
                marker.action = visualization_msgs::msg::Marker::ADD;

                // Set text position (near the object)
                marker.pose.position.x = track.x_pred[0];
                marker.pose.position.y = track.x_pred[1];
                marker.pose.position.z = 2.0; // Raise text above the object

                // Orientation (identity quaternion)
                marker.pose.orientation.x = 0.0;
                marker.pose.orientation.y = 0.0;
                marker.pose.orientation.z = 0.0;
                marker.pose.orientation.w = 1.0;

                marker.text = std::to_string(track.track_id);
                marker.scale.z = 0.7;
                marker.color.r = 1.0;
                marker.color.g = 1.0;
                marker.color.b = 1.0;
                marker.color.a = 1.0;
                if (track.ttc < ttc_upper_limit)
                {
                    marker.text += ", ttc: " + std::to_string(track.ttc) + "s";
                    marker.color.r = 1.0;
                    marker.color.g = 0.0;
                    marker.color.b = 0.0;
                }
                marker.lifetime = rclcpp::Duration(2, 500000000);

                markers.markers.push_back(marker);
            }
        }
        marker_publisher->publish(markers);
    }

    void KF::publishTracks(builtin_interfaces::msg::Time cur_stamp)
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
            yawangle_track = std::atan2(track.x_pred[3], track.x_pred[2]);
            track_obj.pose.orientation.x = 0.0;
            track_obj.pose.orientation.y = 0.0;
            track_obj.pose.orientation.z = std::sin(yawangle_track / 2);
            track_obj.pose.orientation.w = std::cos(yawangle_track / 2);
            track_obj.v_x = track.x_pred[2];
            track_obj.v_y = track.x_pred[3];
            track_obj.ttc = track.ttc;
            track_obj.yawrate = 0.0; // cannot be determined for linear KF
            track_obj.id = track.track_id;
            track_obj.age = track.track_age;
            track_obj.cycles_wo_measurement = track.track_wo_measurement;
            tracklist_obj.tracks.push_back(track_obj);
        }
        pub_tracklist->publish(tracklist_obj);
    }

    void KF::calcTTC(std::vector<Track> &cur_tracks, float v_ego, float ttc_upper_limit)
    {
        for (auto &track : cur_tracks)
        {
            track.ttc = std::numeric_limits<float>::infinity();
            if (abs(track.x_pred[1]) < 1.1 + 0.6)
            {
                track.ttc = (track.x_pred[0] - 3.9) / (v_ego + 1e-6);
                if (track.ttc < 0 || track.ttc > ttc_upper_limit)
                {
                    track.ttc = std::numeric_limits<float>::infinity();
                }
            }
        }
    }
} // namespace cluster_track_cpp

// Register the component with class_loader.
// This acts as a sort of entry point, allowing the component to be discoverable
// when its library is being loaded into a running process.
RCLCPP_COMPONENTS_REGISTER_NODE(cluster_track_cpp::KF)