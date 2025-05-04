#include <chrono>
#include <iostream>
#include <memory>
#include <set>
#include <vector>
#include <fstream>
#include <sstream>
#include <cstdio>

#include <Eigen/Dense>
#include <open3d/Open3D.h>

#include <rspd/planedetector.h>

using namespace open3d;

void output_json(const std::set<Plane*>& planes) {
    std::cout << "[" << std::endl;
    
    auto it = planes.begin();
    while (it != planes.end()) {
        Plane* p = *it;
        std::cout << "  {" << std::endl;
        std::cout << "    \"normal\": [" << p->normal().x() << ", " << p->normal().y() << ", " << p->normal().z() << "]," << std::endl;
        std::cout << "    \"center\": [" << p->center().x() << ", " << p->center().y() << ", " << p->center().z() << "]," << std::endl;
        std::cout << "    \"basisU\": [" << p->basisU().x() << ", " << p->basisU().y() << ", " << p->basisU().z() << "]," << std::endl;
        std::cout << "    \"basisV\": [" << p->basisV().x() << ", " << p->basisV().y() << ", " << p->basisV().z() << "]," << std::endl;
        std::cout << "    \"distanceFromOrigin\": " << p->distanceFromOrigin() << "," << std::endl;
        std::cout << "    \"inlierCount\": " << p->inliers().size() << std::endl;
        std::cout << "  }";
        
        ++it;
        if (it != planes.end()) {
            std::cout << "," << std::endl;
        } else {
            std::cout << std::endl;
        }
    }
    
    std::cout << "]" << std::endl;
}

void print_usage(const char* program_name) {
    std::cerr << "Usage: " << program_name << " <width> <height> [options]" << std::endl;
    std::cerr << "Options:" << std::endl;
    std::cerr << "  --min-normal-diff <degrees>     Minimum normal difference (default: 60)" << std::endl;
    std::cerr << "  --max-dist <degrees>            Maximum distance (default: 75)" << std::endl;
    std::cerr << "  --outlier-ratio <ratio>         Maximum outlier ratio (default: 0.75)" << std::endl;
    std::cerr << "  --min-num-points <count>        Minimum number of points (default: 30)" << std::endl;
    std::cerr << "  --nr-neighbors <count>          Number of neighbors for KNN (default: 75)" << std::endl;
}

int main(int argc, char *argv[]) {
    // Parse width and height from command line arguments
    if (argc < 3) {
        print_usage(argv[0]);
        return 1;
    }
    
    int width = std::stoi(argv[1]);
    int height = std::stoi(argv[2]);
    int totalPoints = width * height;
    
    // Default parameter values
    double minNormalDiff = 60.0;    // degrees
    double maxDist = 75.0;          // degrees
    double outlierRatio = 0.75;     // ratio
    size_t minNumPoints = 30;       // count
    int nrNeighbors = 75;           // count
    
    // Parse optional parameters
    for (int i = 3; i < argc; i++) {
        std::string arg = argv[i];
        
        if (arg == "--min-normal-diff" && i + 1 < argc) {
            minNormalDiff = std::stod(argv[++i]);
        } else if (arg == "--max-dist" && i + 1 < argc) {
            maxDist = std::stod(argv[++i]);
        } else if (arg == "--outlier-ratio" && i + 1 < argc) {
            outlierRatio = std::stod(argv[++i]);
        } else if (arg == "--min-num-points" && i + 1 < argc) {
            minNumPoints = std::stoi(argv[++i]);
        } else if (arg == "--nr-neighbors" && i + 1 < argc) {
            nrNeighbors = std::stoi(argv[++i]);
        } else {
            std::cerr << "Unknown parameter: " << arg << std::endl;
            print_usage(argv[0]);
            return 1;
        }
    }
    
    // Read binary f32 points from stdin
    std::vector<Eigen::Vector3d> points;
    points.reserve(totalPoints);
    
    float point_buffer[3];
    for (int i = 0; i < totalPoints; i++) {
        // Read 3 floats (12 bytes) for each point
        if (fread(point_buffer, sizeof(float), 3, stdin) != 3) {
            std::cerr << "Error reading point " << i << std::endl;
            return 1;
        }
        points.push_back(Eigen::Vector3d(point_buffer[0], point_buffer[1], point_buffer[2]));
    }
    
    // Create Open3D point cloud
    auto cloud_ptr = std::make_shared<geometry::PointCloud>();
    for (const auto& point : points) {
        cloud_ptr->points_.push_back(point);
    }
    
    // Parameters
    const geometry::KDTreeSearchParam &search_param = geometry::KDTreeSearchParamKNN(nrNeighbors);
    
    // Estimate normals
    cloud_ptr->EstimateNormals(search_param);
    
    // Build KD tree for neighbor search
    geometry::KDTreeFlann kdtree;
    kdtree.SetGeometry(*cloud_ptr);
    std::vector<std::vector<int>> neighbors;
    neighbors.resize(cloud_ptr->points_.size());
    
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < (int)cloud_ptr->points_.size(); i++) {
        std::vector<int> indices;
        std::vector<double> distance2;
        if (kdtree.Search(cloud_ptr->points_[i], search_param, indices, distance2)) {
            neighbors[i] = indices;
        }
    }
    
    // Detect planes
    PlaneDetector rspd(cloud_ptr, neighbors, minNumPoints);
    
    // Set optional parameters if provided
    rspd.minNormalDiff(minNormalDiff);
    rspd.maxDist(maxDist);
    rspd.outlierRatio(outlierRatio);
    
    std::set<Plane*> planes = rspd.detect();
    
    // Output planes as JSON
    output_json(planes);
    
    return 0;
}