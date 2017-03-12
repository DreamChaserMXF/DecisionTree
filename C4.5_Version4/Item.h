#ifndef ITEM_H
#define ITEM_H
using std::string;
using std::vector;
using std::map;

class Item
{
public:
	Item(int attr_count, bool isLabeled);
	string	id_;
	vector<double> value_;
	string real_type_;					// 实际类别
	string predicted_type_;				// 预测类别(只能由分类器赋值)
	bool isLabeled_;					// 该Item是否有过标注
};
#endif