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
	string real_type_;					// ʵ�����
	string predicted_type_;				// Ԥ�����(ֻ���ɷ�������ֵ)
	bool isLabeled_;					// ��Item�Ƿ��й���ע
};
#endif