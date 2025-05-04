#include <chrono>
#include <iostream>
#include <memory>
#include <set>
#include <vector>
#include <fstream>
#include <sstream>
#include <cstdio>
#include <algorithm>   // For std::sort

#include <Eigen/Dense>
#include <open3d/Open3D.h>

#include <rspd/planedetector.h>

using namespace open3d;

void output_json(const std::set<Plane*>& planes_set) {
    // Convert set to vector for sorting
    std::vector<Plane*> planes(planes_set.begin(), planes_set.end());
    
    // Log before sorting
    std::cerr << "Planes before sorting:" << std::endl;
    for (size_t i = 0; i < planes.size(); i++) {
        std::cerr << "  Plane " << i << ": " << planes[i]->inliers().size() << " inliers" << std::endl;
    }
    
    // Sort planes by inlier count (descending order)
    std::sort(planes.begin(), planes.end(), [](const Plane* a, const Plane* b) {
        return a->inliers().size() > b->inliers().size();
    });
    
    // Log after sorting
    std::cerr << "Planes after sorting (descending by inlier count):" << std::endl;
    for (size_t i = 0; i < planes.size(); i++) {
        std::cerr << "  Plane " << i << ": " << planes[i]->inliers().size() << " inliers" << std::endl;
    }
    
    // Output sorted planes as JSON
    std::cout << "[" << std::endl;
    
    for (size_t i = 0; i < planes.size(); i++) {
        Plane* p = planes[i];
        std::cout << "  {" << std::endl;
        std::cout << "    \"normal\": [" << p->normal().x() << ", " << p->normal().y() << ", " << p->normal().z() << "]," << std::endl;
        std::cout << "    \"center\": [" << p->center().x() << ", " << p->center().y() << ", " << p->center().z() << "]," << std::endl;
        std::cout << "    \"basisU\": [" << p->basisU().x() << ", " << p->basisU().y() << ", " << p->basisU().z() << "]," << std::endl;
        std::cout << "    \"basisV\": [" << p->basisV().x() << ", " << p->basisV().y() << ", " << p->basisV().z() << "]," << std::endl;
        std::cout << "    \"distanceFromOrigin\": " << p->distanceFromOrigin() << "," << std::endl;
        std::cout << "    \"inlierCount\": " << p->inliers().size() << std::endl;
        std::cout << "  }";
        
        if (i < planes.size() - 1) {
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
    
    // Debug output: show received command-line arguments
    std::cerr << "Command-line arguments received:" << std::endl;
    for (int i = 0; i < argc; i++) {
        std::cerr << "  argv[" << i << "]: " << argv[i] << std::endl;
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
    
    // Track which parameters were provided
    bool minNormalDiffProvided = false;
    bool maxDistProvided = false;
    bool outlierRatioProvided = false;
    bool minNumPointsProvided = false;
    bool nrNeighborsProvided = false;
    
    // Parse optional parameters
    for (int i = 3; i < argc; i++) {
        std::string arg = argv[i];
        
        if (arg == "--min-normal-diff" && i + 1 < argc) {
            minNormalDiff = std::stod(argv[++i]);
            minNormalDiffProvided = true;
        } else if (arg == "--max-dist" && i + 1 < argc) {
            maxDist = std::stod(argv[++i]);
            maxDistProvided = true;
        } else if (arg == "--outlier-ratio" && i + 1 < argc) {
            outlierRatio = std::stod(argv[++i]);
            outlierRatioProvided = true;
        } else if (arg == "--min-num-points" && i + 1 < argc) {
            minNumPoints = std::stoi(argv[++i]);
            minNumPointsProvided = true;
        } else if (arg == "--nr-neighbors" && i + 1 < argc) {
            nrNeighbors = std::stoi(argv[++i]);
            nrNeighborsProvided = true;
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
    
    // Use default number of neighbors (75) unless explicitly provided
    const int knnNeighbors = nrNeighborsProvided ? nrNeighbors : 75;
    const geometry::KDTreeSearchParam &search_param = geometry::KDTreeSearchParamKNN(knnNeighbors);
    
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
    
    // Create PlaneDetector with appropriate minNumPoints
    size_t effectiveMinNumPoints = minNumPointsProvided ? minNumPoints : 30;
    PlaneDetector rspd(cloud_ptr, neighbors, effectiveMinNumPoints);
    
    // Debug output for parameters
    std::cerr << "Point cloud has " << cloud_ptr->points_.size() << " points" << std::endl;
    std::cerr << "Parameters used:" << std::endl;
    std::cerr << "  minNumPoints: " << effectiveMinNumPoints 
              << (minNumPointsProvided ? " (user-provided)" : " (default)") << std::endl;
    std::cerr << "  nrNeighbors: " << knnNeighbors 
              << (nrNeighborsProvided ? " (user-provided)" : " (default)") << std::endl;
    
    // Only set parameters that were explicitly provided
    // Set and log parameters with safer handling
    if (minNormalDiffProvided) {
        rspd.minNormalDiff(minNormalDiff);
        std::cerr << "  minNormalDiff: " << minNormalDiff << " (user-provided)" << std::endl;
    } else {
        // Don't try to access the getter if it might not exist
        std::cerr << "  minNormalDiff: using default value" << std::endl;
    }
    
    if (maxDistProvided) {
        rspd.maxDist(maxDist);
        std::cerr << "  maxDist: " << maxDist << " (user-provided)" << std::endl;
    } else {
        std::cerr << "  maxDist: using default value" << std::endl;
    }
    
    if (outlierRatioProvided) {
        rspd.outlierRatio(outlierRatio);
        std::cerr << "  outlierRatio: " << outlierRatio << " (user-provided)" << std::endl;
    } else {
        std::cerr << "  outlierRatio: using default value" << std::endl;
    }
    
    // Detect planes
    auto start_time = std::chrono::high_resolution_clock::now();
    std::set<Plane*> planes = rspd.detect();
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
    
    // Debug output for results
    std::cerr << "Plane detection completed in " << duration << " ms" << std::endl;
    std::cerr << "Detected " << planes.size() << " planes" << std::endl;
    
    // Output planes as JSON
    output_json(planes);
    
    return 0;
}