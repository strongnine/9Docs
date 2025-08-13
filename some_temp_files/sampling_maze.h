#include <opencv2/opencv.hpp>
#include <vector>

#ifndef SAMPLING_MAZE_H_
#define SAMPLING_MAZE_H_

#define SAFE_DELETE(P)\
{\
if(P)\
delete[](P);\
(P)=nullptr;\
}

#define OBS_COST (-(FLT_MAX / 2))
#define MAX_COST (FLT_MAX / 2)

enum ScaterTypes {
CIRCLE = 1,
RECTANGLE = 2,
BOUNDARY = 3,
};

struct Node {
Node() : x(0.f), y(0.f), parent(nullptr) {};
Node(Node& node):x(node.x), y(node.y), parent(node.parent) {};
Node(const Node& node) {
x = node.x;
y = node.y;
if (node.parent != nullptr) {
parent = new Node(*node.parent);  // deep-copy parent
} else {
parent = nullptr;
};
Node(float x, float y) : x (x), y(y), parent(nullptr) {};
Node(float x, float y, Node* parent) : x(x), y(y), parent(parent) {};
Node operator=(Node* node);
Node& operator=(const Node& node);

bool operator==(const Node& p) const {
  return this->x == p.x && this->y == p.y;
}

bool operator!=(const Node& p) const {
  return this->x != p.x || this->y !=p.y;
}

Node operator-(const Node& p) const {
  return Node(this->x - p.x, this->y - p.y);
}

Node operator+(const Node& p) const {
  return Node(this->x + p.x, this->y + p.y);
}

float dot(const Node& p) const { return this->x * p.x + this->y * p.y; }

float cross(const Node& p) const { return this->x * p.y - p.x * this->y; }

float x;
float y;
Node* parent;
};




