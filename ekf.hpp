// #define PCL_NO_PRECOMPILE // Required to include PCL Headers for custom
// Points
#ifndef CLUSTERTRACK_CPP__EKF_NODE_HPP_
#define CLUSTERTRACK_CPP__EKF_NODE_HPP_

#include <Eigen/Dense>
#include <vector>
#include <unordered_map>
#include <chrono>

// for message synchronization
#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/synchronizer.h>

// ROS includes
#include <flexdev_interfaces/msg/eml01.hpp>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <rclcpp/node.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <track_msgs/msg/track.hpp>
#include <track_msgs/msg/track_list.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include <visualization_msgs/msg/marker_array.hpp>

// include the specific radar data type
#include "radar_data/pointRadarRawCrit2.h"
#include "cluster_track_cpp/visibility_control.h"

namespace cluster_track_cpp
{

    typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::msg::PointCloud2, sensor_msgs::msg::PointCloud2>
        SyncPolicyT;

    typedef PointRadarRawCrit2 PointTCrit;
    typedef sensor_msgs::msg::PointCloud2 PointTCritMsg;

    class EKF : public rclcpp::Node
    {
    public:
        COMPOSITION_PUBLIC
        explicit EKF(const rclcpp::NodeOptions &options);
        virtual ~EKF() {}
        /**
         *  Initialize the publisher, subscribers, timers
         *  and parameters from the yaml files
         */
        void init();
        /**
         */

    private:
        bool first_run = true;
        std::string centroids_in_topic, cloud_in_topic, cloud_out_topic, centroids_out_topic;
        int wo_measurement_age;
        double association_distance, dont_trust_factor, prediction_error_factor, measurement_error_factor;
        float cycle_time, epsilon;
        std::string ekf_coordinate_system;

        long int cur_timestamp, last_timestamp;

        // struct declaration
        struct Track
        {
            Eigen::Matrix<double, 5, 1> x_pred; // state vector, [x, y, v, yaw, yawrate]
            Eigen::Matrix<double, 5, 5> P_pred; // covariance matrix 5x5
            int track_id;
            int track_age;            // total age of track in cycles
            int track_wo_measurement; // number of track cycles without measurement
        };

        // subscriber
        rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr clustered_cloud_subscriber;
        rclcpp::Subscription<flexdev_interfaces::msg::EML01>::SharedPtr eml01_subscriber;

        // Function declarations
        void ekfTracking(const sensor_msgs::msg::PointCloud2::SharedPtr clustered_cloud_msg);
        void eml01_callback(const flexdev_interfaces::msg::EML01::SharedPtr eml01_msg);
        void predict(Track &track, double dt);
        void update(Track &track, const pcl::PointXYZ &measurement);
        void publishMarkers();
        void publishTracks(builtin_interfaces::msg::Time cur_stamp);
        void deleteTracks(std::vector<Track> &cur_tracks, const int wo_measurement_age = 5);
        Eigen::MatrixXd createAssociation(const std::vector<Track> cur_tracks, 
            const pcl::PointCloud<pcl::PointXYZ>::Ptr pcl_centroids, float non_assignment_cost);
        
        // vector which contains all tracks
        std::vector<Track> cur_tracks;
        std::vector<Eigen::Vector2d> measurements;

        // custom track and track list
        track_msgs::msg::Track track_obj;
        track_msgs::msg::TrackList tracklist_obj;

        // variables for prediction
        double dt;
        double x_pos, y_pos, v, yaw, yawd;
        double x_centroid, y_centroid, vdop_centroid;
        float v_ego;

        // variables for hungarian matching
        unsigned int num_tracks, num_detections, num_dummy_tracks, num_dummy_detections;
        double cost;
        Eigen::Matrix<double, 1, 1> cost_matrix;
        Eigen::Vector2d pos_meas, pos_pred;

        // Eigen matrices
        Eigen::Matrix<double, 5, 5> F;
        Eigen::Matrix<double, 2, 5> H;
        // Eigen::MatrixXd P_init = Eigen::MatrixXd::Identity(5, 5) * 1.0;
        // process noise covariance matrix
        Eigen::Matrix<double, 5, 5> Q;

        // Eigen vectors
        Eigen::Vector2d z, z_pred, y, m;

        // measurement noise covariance matrix
        Eigen::Matrix<double, 2, 2> R;
        Eigen::Matrix<double, 2, 2> S;
        Eigen::Matrix<double, 2, 2> Sinverse;
        Eigen::Matrix<double, 5, 2> K;
        // Eigen::Matrix<double, 5, 5> P_pred; TODO: delete

        // maps for centroid->
        // this map stores the point indices which belong to each centroid
        std::map<int, std::vector<int>> centroid_to_clusterid;
        std::map<int, int> centroid_to_track;

        // points and point clouds
        pcl::PointCloud<PointTCrit> tracked_cloud;
        pcl::PointXYZ object_hypothesis;
        pcl::PointCloud<pcl::PointXYZ> object_hypotheses;

        // variables for association funciton
        // size_t num_tracks, num_detections, num_dummy_tracks, num_dummy_detections;
        // Eigen::MatrixXd cost_matrix;
        // Eigen::Matrix<double, 2, 1> pos_pred, pos_meas;
        // double cost;

        // messages
        sensor_msgs::msg::PointCloud2 output_cloud;
        sensor_msgs::msg::PointCloud2 output_centroids;

        // RViz2 objects
        visualization_msgs::msg::Marker marker;
        visualization_msgs::msg::MarkerArray markers;

        // publisher
        rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_centroids;
        rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_cloud_out;
        rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr marker_publisher;
        rclcpp::Publisher<track_msgs::msg::TrackList>::SharedPtr pub_tracklist;
    };

} // namespace cluster_track_cpp

#endif