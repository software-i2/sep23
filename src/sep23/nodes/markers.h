// Copyright by BeeX [2026]
#pragma once

#include <sep23/collision.h>

#include <sensor_msgs/PointCloud2.h>
#include <tf2_eigen/tf2_eigen.h>
#include <sensor_msgs/point_cloud2_iterator.h>
#include <visualization_msgs/MarkerArray.h>

namespace sep23 {

inline geometry_msgs::Point toPoint(const Eigen::Vector3d &v) {
    geometry_msgs::Point p;
    p.x = v.x();
    p.y = v.y();
    p.z = v.z();
    return p;
}

inline visualization_msgs::Marker marker(const std::string &frame, const std::string &ns, int type, float r, float g, float b,
                                         float a, double size) {
    visualization_msgs::Marker m;
    m.header.frame_id    = frame;
    m.header.stamp       = ros::Time::now();
    m.ns                 = ns;
    m.type               = type;
    m.action             = visualization_msgs::Marker::ADD;
    m.pose.orientation.w = 1.0;
    m.scale.x = m.scale.y = m.scale.z = size;
    m.color.r = r;
    m.color.g = g;
    m.color.b = b;
    m.color.a = a;
    return m;
}

// Occupied cells as points in `frame`, with intensity 1 on handle cells.
inline sensor_msgs::PointCloud2 mapCloud(const ObstacleMap &map, const Eigen::Isometry3d &map_to_frame, const std::string &frame) {
    sensor_msgs::PointCloud2 cloud;
    cloud.header.frame_id = frame;
    cloud.header.stamp    = ros::Time::now();
    sensor_msgs::PointCloud2Modifier modifier(cloud);
    modifier.setPointCloud2Fields(4, "x", 1, sensor_msgs::PointField::FLOAT32, "y", 1, sensor_msgs::PointField::FLOAT32, "z", 1,
                                  sensor_msgs::PointField::FLOAT32, "intensity", 1, sensor_msgs::PointField::FLOAT32);
    modifier.resize(map.obstacle.size() + map.handle.size());
    sensor_msgs::PointCloud2Iterator<float> x(cloud, "x"), y(cloud, "y"), z(cloud, "z"), i(cloud, "intensity");
    for (const std::vector<uint32_t> *cells : {&map.obstacle, &map.handle}) {
        for (const uint32_t c : *cells) {
            const Eigen::Vector3d p = map_to_frame * map.box.centre(c);
            *x = static_cast<float>(p.x());
            *y = static_cast<float>(p.y());
            *z = static_cast<float>(p.z());
            *i = cells == &map.handle ? 1.0f : 0.0f;
            ++x;
            ++y;
            ++z;
            ++i;
        }
    }
    return cloud;
}

// Link capsules and blade samples of the collision body.
inline visualization_msgs::MarkerArray bodyMarkers(const Body &body, double link_radius, const std::string &frame) {
    visualization_msgs::MarkerArray out;
    int                             id = 0;
    for (const Segment &s : body.links) {
        visualization_msgs::Marker tube = marker(frame, "links", visualization_msgs::Marker::CYLINDER, 0.2f, 0.6f, 1.0f, 0.35f, 2.0 * link_radius);
        const Eigen::Vector3d      span = s.b - s.a;
        const Eigen::Quaterniond   turn = Eigen::Quaterniond::FromTwoVectors(Eigen::Vector3d::UnitZ(), span.norm() > 0.0 ? span : Eigen::Vector3d::UnitZ());
        tube.id                         = id++;
        tube.pose.position              = toPoint(0.5 * (s.a + s.b));
        tube.pose.orientation   = tf2::toMsg(turn);
        tube.scale.z            = span.norm();
        out.markers.push_back(tube);
    }
    visualization_msgs::Marker blades = marker(frame, "blades", visualization_msgs::Marker::SPHERE_LIST, 0.2f, 0.6f, 1.0f, 0.35f, 0.003);
    for (const Eigen::Vector3d &p : body.blades) {
        blades.points.push_back(toPoint(p));
    }
    out.markers.push_back(blades);
    return out;
}

// The hull footprint as a flat plate at its floor height.
inline visualization_msgs::Marker hullMarker(const Hull &h, const std::string &frame) {
    visualization_msgs::Marker plate = marker(frame, "hull", visualization_msgs::Marker::TRIANGLE_LIST, 0.8f, 0.3f, 0.3f, 0.3f, 1.0);
    const Eigen::Vector3d      c[4]  = {{h.min_x, h.min_y, h.floor_z}, {h.max_x, h.min_y, h.floor_z}, {h.max_x, h.max_y, h.floor_z},
                                        {h.min_x, h.max_y, h.floor_z}};
    for (const int k : {0, 1, 2, 0, 2, 3}) {
        plate.points.push_back(toPoint(c[k]));
    }
    return plate;
}

}  // namespace sep23
