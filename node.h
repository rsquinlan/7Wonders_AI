#pragma once
#include <memory>
#include <algorithm>
#include <game.h>
#include <card.h>
#include <iostream>

class Node
{
private:
	DMAG::Game state;
	std::shared_ptr<Node> parent = nullptr;
	std::vector<std::unique_ptr<Node>> children;

	int visitCount;
	int value;

public:
	Node(DMAG::Game state, std::shared_ptr<Node> parent = nullptr);
	void addChild(std::unique_ptr<Node> child);
	std::shared_ptr<Node> selectBestChild() const;
	void update(double value);
	DMAG::Game getState() const;
	std::shared_ptr<Node> getParent() const;
	const std::vector<std::unique_ptr<Node>>& getChildren() const;
	int getVisitCount() const;
	int getValue() const;
	bool isLeaf() const;
	void expand();
};