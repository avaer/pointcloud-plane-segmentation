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

int main(int argc, char *argv[]) {
    // Parse width and height from command line arguments
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <width> <height>" << std::endl;
        return 1;
    }
    
    int width = std::stoi(argv[1]);
    int height = std::stoi(argv[2]);
    int totalPoints = width * height;
    
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
    static constexpr int nrNeighbors = 75;
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
    PlaneDetector rspd(cloud_ptr, neighbors);
    std::set<Plane*> planes = rspd.detect();
    
    // Output planes as JSON
    output_json(planes);
    
    return 0;
}