// Copyright by BeeX [2026]
// Stage 3 benchmark, not part of the pick. A clean twin of the production tracker runs on /image and /pointcloud while the
// pick's own tracker runs on the occluded camera. Both use /pick/track/* as loaded, are seeded on the same frame from the
// pick's first /tracker/target_pose, and are compared stamp by stamp: the difference is the error the occlusion causes.
// One CSV row per frame to ~out. Run next to: pick.launch track:=true occluded:=true, and exp/occluder.py _follow_pick:=true
#include <sep23/PickState.h>
#include <sep23/track.h>

#include <geometry_msgs/PoseStamped.h>
#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <opencv2/imgproc.hpp>
#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/point_cloud2_iterator.h>

#include <cmath>
#include <deque>
#include <fstream>
#include <map>
#include <mutex>

using namespace sep23;

namespace {

double number(const std::string &key) {
    double v = 0.0;
    if (!ros::param::get(key, v)) {
        throw std::runtime_error(key + " is not set: start pick.launch first");
    }
    return v;
}

std::vector<Eigen::Vector3f> points(const sensor_msgs::PointCloud2 &cloud) {
    std::vector<Eigen::Vector3f> out;
    out.reserve(static_cast<size_t>(cloud.width) * cloud.height);
    sensor_msgs::PointCloud2ConstIterator<float> x(cloud, "x"), y(cloud, "y"), z(cloud, "z");
    for (; x != x.end(); ++x, ++y, ++z) {
        out.emplace_back(*x, *y, *z);
    }
    return out;
}

struct Frame {
    ros::Time                    stamp;
    cv::Mat                      gray;
    std::vector<Eigen::Vector3f> cloud;
};

struct Row {
    bool            have_occluded = false, have_clean = false, clean_ok = false;
    Eigen::Vector3d occluded = Eigen::Vector3d::Zero(), clean = Eigen::Vector3d::Zero();
    double          coverage = -1.0;
};

class Bench {
public:
    Bench(ros::NodeHandle &nh, const std::string &out, int mode)
            : mode_(mode),
              clean_(settings(), camera()),
              image_sub_(nh, "/image", 2),
              cloud_sub_(nh, "/pointcloud", 2),
              raw_sync_(image_sub_, cloud_sub_, 5),
              csv_(out, std::ios::app) {
        if (csv_.tellp() == 0) {
            csv_ << "mode,run,t_s,coverage,occ_x,occ_y,occ_z,clean_x,clean_y,clean_z,err_mm,clean_ok\n";
        }
        raw_sync_.registerCallback(&Bench::onRaw, this);
        occluded_cloud_ = nh.subscribe("/occluder/pointcloud", 2, &Bench::onOccludedCloud, this);  // the feed the pick reads
        pose_  = nh.subscribe("/tracker/target_pose", 20, &Bench::onPose, this);
        state_ = nh.subscribe("/pick/state", 10, &Bench::onState, this);
    }

private:
    static TrackSettings settings() {
        return {number("/pick/track/depth_gate_m"), number("/pick/track/roi_radius_m"), static_cast<int>(number("/pick/track/min_points")),
                static_cast<int>(number("/pick/track/max_points")), number("/pick/track/ransac_px")};
    }
    static CameraModel camera() {
        const std::string k = "/robot/camera/intrinsics/";
        return {static_cast<int>(number(k + "width_px")), static_cast<int>(number(k + "height_px")), number(k + "fx_px"),
                number(k + "fy_px"), number(k + "cx_px"), number(k + "cy_px")};
    }

    // A run is one blind motion: GOTOGRASP opens it, leaving CLOSEJAW writes it out.
    void onState(const PickState::ConstPtr &msg) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (msg->state == "GOTOGRASP" && !armed_) {
            ++run_, armed_ = true, seeded_ = false;
            rows_.clear();
        } else if (armed_ && msg->state != "GOTOGRASP" && msg->state != "CLOSEJAW") {
            flush();
            armed_ = seeded_ = false;
            clean_.stop();
        }
    }

    // The pick's first pose of a run is its start target, published on its seed frame: the twin starts from both.
    void onPose(const geometry_msgs::PoseStamped::ConstPtr &msg) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!armed_) {
            return;
        }
        const Eigen::Vector3d p(msg->pose.position.x, msg->pose.position.y, msg->pose.position.z);
        Row &row = rows_[msg->header.stamp.toNSec()];
        row.have_occluded = true, row.occluded = p;
        if (!seeded_) {
            clean_.start(p);
            seeded_ = true, seed_stamp_ = msg->header.stamp;
            // Replay from the seed frame on, in arrival order: after a bag loop, stamps restart and cannot be compared.
            bool from_seed = false;
            for (const Frame &f : recent_) {
                from_seed = from_seed || f.stamp == seed_stamp_;
                if (from_seed) {
                    track(f);
                }
            }
            recent_.clear();
        }
    }

    void onRaw(const sensor_msgs::Image::ConstPtr &image, const sensor_msgs::PointCloud2::ConstPtr &cloud) {
        if (image->encoding != "bgr8") {
            ROS_WARN_THROTTLE(5.0, "[stage3_bench] expected bgr8, got %s", image->encoding.c_str());
            return;
        }
        Frame f{image->header.stamp, cv::Mat(), points(*cloud)};
        cv::cvtColor(cv::Mat(static_cast<int>(image->height), static_cast<int>(image->width), CV_8UC3, const_cast<uint8_t *>(image->data.data()),
                             image->step),
                     f.gray, cv::COLOR_BGR2GRAY);
        std::lock_guard<std::mutex> lock(mutex_);
        if (seeded_) {
            track(f);
        } else {
            recent_.push_back(std::move(f));
            while (recent_.size() > 5) {
                recent_.pop_front();
            }
        }
    }

    void onOccludedCloud(const sensor_msgs::PointCloud2::ConstPtr &cloud) {
        size_t                                       hidden = 0, total = 0;
        sensor_msgs::PointCloud2ConstIterator<float> z(*cloud, "z");
        for (; z != z.end(); ++z, ++total) {
            hidden += !std::isfinite(*z);
        }
        std::lock_guard<std::mutex> lock(mutex_);
        if (armed_ && total > 0) {
            rows_[cloud->header.stamp.toNSec()].coverage = static_cast<double>(hidden) / total;  // the raw bag cloud has no NaNs
        }
    }

    void track(const Frame &f) {
        const TrackResult r = clean_.update(f.gray, f.cloud);
        Row &row = rows_[f.stamp.toNSec()];
        row.have_clean = true, row.clean = r.target, row.clean_ok = r.ok;
    }

    void flush() {
        for (const auto &[stamp, row] : rows_) {
            if (!row.have_occluded || !row.have_clean || row.coverage < 0.0) {
                continue;
            }
            const double t = (ros::Time().fromNSec(stamp) - seed_stamp_).toSec();
            csv_ << mode_ << ',' << run_ << ',' << t << ',' << row.coverage << ',' << row.occluded.x() << ',' << row.occluded.y() << ','
                 << row.occluded.z() << ',' << row.clean.x() << ',' << row.clean.y() << ',' << row.clean.z() << ','
                 << 1000.0 * (row.occluded - row.clean).norm() << ',' << row.clean_ok << '\n';
        }
        csv_.flush();
        ROS_INFO("[stage3_bench] run %d written: %zu frames", run_, rows_.size());
    }

    int                   mode_;
    std::mutex            mutex_;
    Tracker               clean_;
    bool                  armed_ = false, seeded_ = false;
    int                   run_   = 0;
    ros::Time             seed_stamp_;
    std::deque<Frame>     recent_;
    std::map<uint64_t, Row> rows_;

    message_filters::Subscriber<sensor_msgs::Image>                                  image_sub_;
    message_filters::Subscriber<sensor_msgs::PointCloud2>                            cloud_sub_;
    message_filters::TimeSynchronizer<sensor_msgs::Image, sensor_msgs::PointCloud2> raw_sync_;
    ros::Subscriber                                                                  occluded_cloud_, pose_, state_;
    std::ofstream                                                                    csv_;
};

}  // namespace

int main(int argc, char **argv) {
    ros::init(argc, argv, "stage3_bench");
    ros::NodeHandle nh, pnh("~");
    std::string     out  = pnh.param<std::string>("out", "stage3_bench.csv");
    const int       mode = pnh.param("mode", -1);
    while (ros::ok() && !ros::param::has("/pick/track/depth_gate_m")) {
        ros::Duration(0.5).sleep();  // started alongside pick.launch: its parameters load a moment later
    }
    try {
        Bench bench(nh, out, mode);
        ros::AsyncSpinner spinner(2);
        spinner.start();
        ros::waitForShutdown();
    } catch (const std::runtime_error &e) {
        ROS_ERROR("[stage3_bench] %s", e.what());
        return 1;
    }
    return 0;
}
