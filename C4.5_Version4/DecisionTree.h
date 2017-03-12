#ifndef DECISIONTREE_H
#define DECISIONTREE_H

#include <iostream>
#include <vector>
#include <map>
using std::vector;
using std::ostream;



//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// DecisionTree��
// ���ýӿڣ�
//  ���캯����
//		������
//			1��
//				items			����������
//				attr_array		��������
//				positive_type	Ŀ�����
//			2��
//				filename		�����������ڵ��ļ���
//				attr_array		���ԣ�����������
//				positive_type	Ŀ�����
//		���ã�
//			����������(ֱ�ӻ���ļ���)����DecisionTree�����趨���Ա�ǩ��Ŀ����𣨷�����Ҫ�������
//
//	Train()
//		���ã�
//			��������ѵ����������
//
//	SelfCheck()
//		���ã�
//			������������������������
//
//	Detect(const vector<Item> &items)
//		������
//			ʵ������
//		���ã�
//			��itemsʵ�����з���
//	Detect(const char *filename);
//		������
//			ʵ���������ڵ��ļ���
//		���ã�
//			�����ļ��е�ʵ������������
//
//	Print()
//	Print(const char *filename)
//		����:
//			Ҫ������ļ���
//		���ã�
//			��DecisionTree�����д洢��ʵ�����������̨���ļ�
//
//
//
//	ע�����	
//		1��������û�жԹ��캯���е������б�������������ݵ�һ���Խ��м��飬�ʴ���ʹ��ǰ����ȷ�������б�������������ȫ���
//		2��DecisionTree��������֮�����࣬�����������������쳣
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

	// ��Ա����
	void			Train		(bool isprint);
	void			LoadItem	(const vector<Item> &items, bool IsLabeled);
	void			LoadItem	(const char *filename, bool IsLabeled);
	ConfusionMatrix	Detect		(bool isprint, double prob_thresh = 0.5);	// ����������ΪҶ�ӽ���������ı����ﵽ���ٲ��ж�Ϊ����
	void			CrossValidation(size_t times_vali, double prob_thresh = 0.5);	// ����������ΪҶ�ӽ���������ı����ﵽ���ٲ��ж�Ϊ����


	void			PrintRule	()						const;
	void			Print		()						const;
	void			Print		(const char *filename)	const;

//	DecisionTree& operator = (const DecisionTree &root);

private:
	// ��ʾĳ�����Եķ�����Ϣ�������������������еļ�ֵ������ʱ����ֵ����С�����ʺͷ�����������˵ļ�����
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
		double min_;		// ��ǰ�����������е���Сֵ
		double max_;		// ��ǰ�����������е����ֵ
		vector<double> threshold_;	// ��ֵ
		double max_gainratio_;		// ��Ϣ������
		map<string, int> inside_user_count_;
		map<string, int> outside_user_count_;
		map<string, int> &left_user_count_;
		map<string, int> &right_user_count_;
	};
	
	// ˽�г�Ա����
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

	// ���ݳ�Ա
	vector<DecisionTree>	subTree_;			// �ý�������

	vector<Item>			items_;				// �洢��ǰ����
	bool					islabeled_;			// ��ʾ�����Ƿ񺬱�ǩ
	
	map<string, int>		type_;				// Key��ʾ���࣬ Value��ʾ�������������е�Ƶ��
	int						min_typenumber_;	// �������С����������ʾ��ĳ�������һ������С�ڸ��������ٷ�����
	int						depth_;				// ��ǰ�������
	int				max_depth_;			// ����������(��ʾ�������������������)

	vector<string>			attr_array_;		// �����б�
	vector<bool>			avai_attr_;			// ���õ������б��ò���ֵ��ʾ��
	vector<bool>			used_attr_;			// ʹ�ù�������
	int						cur_attr_;			// ��ǰ�������õķ�������
	vector<double>			threshold_;			// ��ǰ���������������ֵ������ֵʱ����Ϊ1��˫��ֵʱ����Ϊ2��
	string					rule_;				// ÿ����Ҷ�ӽ���Ӧһ������
	
	string					positive_type_;		// Ҫ��������
	string					negative_type_;		// ����������
	THRESH_TYPE				thresh_type_;		// ��ֵ������
	double					leaf_prob_;			// Ҷ�ڵ��������ĸ���
};

ostream& operator << (ostream &out, const DecisionTree::ConfusionMatrix &c_mat);

#endif