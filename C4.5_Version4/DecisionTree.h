#ifndef DECISIONTREE_H
#define DECISIONTREE_H

#include <iostream>
#include <vector>
#include <map>
using std::vector;
using std::ostream;



//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// DecisionTree类
// 公用接口：
//  构造函数：
//		参数：
//			1、
//				items			样本向量组
//				attr_array		特征向量
//				positive_type	目标类别
//			2、
//				filename		样本向量所在的文件名
//				attr_array		属性（特征）向量
//				positive_type	目标类别
//		作用：
//			将样本向量(直接或从文件中)导入DecisionTree对象，设定属性标签，目标类别（分类器要检测的类别）
//
//	Train()
//		作用：
//			利用样本训练出决策树
//
//	SelfCheck()
//		作用：
//			检测自身样本，输出混淆矩阵
//
//	Detect(const vector<Item> &items)
//		参数：
//			实例向量
//		作用：
//			对items实例进行分类
//	Detect(const char *filename);
//		参数：
//			实例向量所在的文件名
//		作用：
//			载入文件中的实例向量并分类
//
//	Print()
//	Print(const char *filename)
//		参数:
//			要输出的文件名
//		作用：
//			将DecisionTree对象中存储的实例输出至控制台或文件
//
//
//
//	注意事项：	
//		1、该类中没有对构造函数中的属性列表参数与样本数据的一致性进行检验，故此在使用前必须确保属性列表与样本数据完全相符
//		2、DecisionTree分类器暂之分两类，多于两类会引起程序异常
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////

class DecisionTree
{
public:
	class ConfusionMatrix
	{
	public:
		ConfusionMatrix() : 
		  true_positive_(0), true_negative_(0),
		false_positive_(0), false_negative_(0)
		{}
		ConfusionMatrix(int true_positive, int true_negative, int false_positive, int false_negative) : 
						true_positive_(true_positive), true_negative_(true_negative),
						false_positive_(false_positive), false_negative_(false_negative)
		{}
		ConfusionMatrix& operator += (const ConfusionMatrix m)
		{
			true_positive_ += m.true_positive_;
			true_negative_ += m.true_negative_;
			false_positive_ += m.false_positive_;
			false_negative_ += m.false_negative_;
			return *this;
		}
		int true_positive_;
		int true_negative_;
		int false_positive_;
		int false_negative_;


	};

	enum THRESH_TYPE{SNG = 1, DBL = 2};

	//DecisionTree(const vector<Item>		&items,
	//			const vector<string>	&attr_array,
	//			const string			&positive_type,
	//			int						min_typenumber,
	//			int						max_depth);
	DecisionTree() : max_depth_(0)
	{}
	DecisionTree(const vector<string>	&attr_array,
				const vector<bool>		&avai_attr,
				const string			&positive_type,
				THRESH_TYPE				thresh_type,
				int						min_typenumber,
				int						max_depth);
	DecisionTree(const vector<Item>		&items,
				const map<string, int>	&type,
				const vector<string>	&attr_array,
				const vector<bool>		&avai_attr,
				const vector<bool>		&used_attr,
				int						cur_attr,
				const string			&positive_type,
				const string			&negative_type,
				THRESH_TYPE				thresh_type,
				int						min_typenumber,
				const string			&rule,
				int						depth,
				int						max_depth);

	~DecisionTree();

	// 成员函数
	void			Train		(bool isprint);
	void			LoadItem	(const vector<Item> &items, bool IsLabeled);
	void			LoadItem	(const char *filename, bool IsLabeled);
	ConfusionMatrix	Detect		(bool isprint, double prob_thresh = 0.5);	// 参数的意义为叶子结点中正例的比例达到多少才判定为正例
	void			CrossValidation(size_t times_vali, double prob_thresh = 0.5);	// 参数的意义为叶子结点中正例的比例达到多少才判定为正例


	void			PrintRule	()						const;
	void			Print		()						const;
	void			Print		(const char *filename)	const;

//	DecisionTree& operator = (const DecisionTree &root);

private:
	// 表示某个属性的分类信息，包括该属性在样本中的极值、分类时的阈值、最小错误率和分类后左右两端的计数器
	class Attr
	{
	public:
		Attr(double min, double max,
			vector<double> threshold,
			double gainratio, 
			const map<string, int> &inside_user_count,
			const map<string, int> &outside_user_count
			) : 			
			min_(min),
			max_(max),
			threshold_(threshold),
			max_gainratio_(gainratio),
			inside_user_count_(inside_user_count),
			outside_user_count_(outside_user_count),
			left_user_count_(inside_user_count_),
			right_user_count_(outside_user_count_)
		{}
		Attr() : 
		min_(0.0),
		max_(0.0),
		threshold_(0),
		max_gainratio_(0.0),
		inside_user_count_(),
		outside_user_count_(),
		left_user_count_(inside_user_count_),
		right_user_count_(outside_user_count_)
		{}
		Attr& operator = (const Attr &attr)
		{
			min_ = attr.min_;
			max_ = attr.max_;
			threshold_ = attr.threshold_;
			max_gainratio_ = attr.max_gainratio_;
			inside_user_count_ = attr.inside_user_count_;
			outside_user_count_ = attr.outside_user_count_;
			return *this;
		}
		double min_;		// 当前属性在样本中的最小值
		double max_;		// 当前属性在样本中的最大值
		vector<double> threshold_;	// 阈值
		double max_gainratio_;		// 信息增益率
		map<string, int> inside_user_count_;
		map<string, int> outside_user_count_;
		map<string, int> &left_user_count_;
		map<string, int> &right_user_count_;
	};
	
	// 私有成员函数
	void	InitialTree();

	bool	ReadSample		(const char *filename, bool IsLabeled);
	void	InitialTypeCount();
	void	ItemSort		(int attr_index);
	void	ItemInOrder		();
	void	ClearItem		();
	double	Info			(map<string, int>type, map<string, int>::size_type size);
	Attr	AttrSNGTrain	(int attr_index, double parent_info);
	Attr	AttrDBLTrain	(int attr_index, double parent_info);
	void	PrintClassInfo	(Attr maxratio_result, const map<string, int> &type_);
	double	Classify		(const Item &item);

	// 数据成员
	vector<DecisionTree>	subTree_;			// 该结点的子树

	vector<Item>			items_;				// 存储当前数据
	bool					islabeled_;			// 表示属性是否含标签
	
	map<string, int>		type_;				// Key表示种类， Value表示该种类在样本中的频数
	int						min_typenumber_;	// 种类的最小个案数，表示若某结点其中一个属性小于该数，则不再分子树
	int						depth_;				// 当前树的深度
	int				max_depth_;			// 树的最大深度(表示用作分类的属性最大个数)

	vector<string>			attr_array_;		// 属性列表
	vector<bool>			avai_attr_;			// 可用的属性列表（用布尔值表示）
	vector<bool>			used_attr_;			// 使用过的属性
	int						cur_attr_;			// 当前子树所用的分类属性
	vector<double>			threshold_;			// 当前属性用作分类的阈值（单阈值时长度为1，双阈值时长度为2）
	string					rule_;				// 每个非叶子结点对应一条规则
	
	string					positive_type_;		// 要检测的种类
	string					negative_type_;		// 正常的种类
	THRESH_TYPE				thresh_type_;		// 阈值的种类
	double					leaf_prob_;			// 叶节点是正例的概率
};

ostream& operator << (ostream &out, const DecisionTree::ConfusionMatrix &c_mat);

#endif