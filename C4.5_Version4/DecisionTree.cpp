#include "stdafx.h"

DecisionTree::DecisionTree(	const vector<string>	&attr_array,
							const vector<bool>		&avai_attr,
							const string			&positive_type,
							THRESH_TYPE				thresh_type,
							int						min_typenumber,
							int						max_depth) :
	subTree_(),
	items_(),	// 函数体中Read
	islabeled_(false),
	type_(),
	attr_array_(attr_array),
	avai_attr_(avai_attr),
	used_attr_(attr_array.size(), false),
	cur_attr_(-1),
//	threshold_(),
	positive_type_(positive_type),
	thresh_type_(thresh_type),
	min_typenumber_(min_typenumber),
	rule_(""),
	depth_(0),
	max_depth_(max_depth)
{}

DecisionTree::DecisionTree(const vector<Item>	&items,
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
						int						max_depth) : 
	subTree_(),
	items_(items),
	islabeled_(true),		// 这句貌似不是必须的，但好歹给人家初始化一下啦
	type_(type),
	attr_array_(attr_array),
	avai_attr_(avai_attr),
	used_attr_(used_attr),
	cur_attr_(-1),	// 初值为-1
	threshold_(static_cast<int>(thresh_type)),
	positive_type_(positive_type),
	negative_type_(negative_type),
	thresh_type_(thresh_type),
	min_typenumber_(min_typenumber),
	rule_(rule),
	depth_(depth + 1),
	max_depth_(max_depth)
{
	// 将父结点的分类属性设为已用
	if(cur_attr < 0)
	{
		cerr << "Error! current attr must be positive!" << endl;
		exit(1);
	}
	else if(true == used_attr[cur_attr])
	{
		cerr << "Error! current attr is used!" << endl;
		exit(1);
	}	
	used_attr_[cur_attr] = true;
}

DecisionTree::~DecisionTree()
{

}

void DecisionTree::PrintClassInfo(Attr result, const map<string, int> &type_)
{
	cout << "\tmin:\t\t" << result.min_ << endl;
	cout << "\tmax:\t\t" << result.max_ << endl;
	if(SNG == thresh_type_)
	{
		cout << "\tthreshold\t" << result.threshold_[0] << endl;
	}
	else
	{
		cout << "\tleft threshold:\t" << result.threshold_[0] << "\tright threshold: " << result.threshold_[1] << endl;
	}
	cout << "\tmax gain ratio:\t" << result.max_gainratio_ << endl;
	// 输出统计表格
	cout << "-----------------------------------------------------" << endl;
	cout << "\t\ttype:";

	// 先输出种类标记
	cout << '\t' << positive_type_ << '\t' << negative_type_ << endl;
	cout << ((SNG == thresh_type_) ? "    left interval:" : "    inside interval:");
	cout << '\t' << result.inside_user_count_[positive_type_] << '\t'<< result.inside_user_count_[negative_type_] << endl;

	cout << ((SNG == thresh_type_) ? "    right interval:" : "    outside interval:");
	cout << '\t' << result.outside_user_count_[positive_type_] << '\t'<< result.outside_user_count_[negative_type_] << endl;

	cout << ((SNG == thresh_type_) ? "\tleft ratio" : "\tinside ratio:");
	cout.precision(2);
	cout << '\t';			
	cout.fill('0');
	cout.width(5);
	cout.setf(ios::left);
	cout << (result.inside_user_count_.count(positive_type_) > 0 ?
		static_cast<double>(result.inside_user_count_.at(positive_type_)) 
		/ static_cast<double>(type_.at(positive_type_)) 
		: 0.0) << '\t';
	cout << (result.inside_user_count_.count(negative_type_) > 0 ?
		static_cast<double>(result.inside_user_count_.at(negative_type_)) 
		/ static_cast<double>(type_.at(negative_type_)) 
		: 0.0) << endl << endl;		
	cout.precision(6);
}

// 用当前所包含的item向量进行训练，确定最佳阈值
void DecisionTree::Train(bool isprint)
{
	if(false == islabeled_)
	{
		cerr << "cannot train without label!" << endl;
		exit(EXIT_FAILURE);
	}
	if(subTree_.size() > 0)
	{
		subTree_.clear();
	}
	if(items_.size() == 0)
	{
		leaf_prob_ = 0.5;
		return;
	}

	vector<Item> temp_item = items_;	//  临时存储items_的排列顺序，Train函数结束后恢复

	int positive = type_[positive_type_];
	int negative = type_[negative_type_];
	if(type_.size() == 2 && positive > min_typenumber_ && negative > min_typenumber_ && depth_ < max_depth_)// 包含足够数量的两种个案且深度没达到最大才分类
	{
		// 先计算当前信息熵
 		double cur_info(0.0);
		for(map<string, int>::const_iterator iter = type_.begin(); iter != type_.end(); ++iter)
		{
			if(iter->second > 0)
			{
				cur_info -= static_cast<double>(iter->second) / static_cast<double>(items_.size())
					* log(static_cast<double>(iter->second) / static_cast<double>(items_.size()));
			}
		}
//		cout << "current information:\t" << cur_info << endl;
		Attr maxratio_result;
		for(vector<bool>::size_type i = 0; i < attr_array_.size(); ++i)
		{
			if(true == avai_attr_[i] && false == used_attr_[i])	// 没用过的属性才可以继续用来分类——对连续变量有必要吗？？？？？？？？？
			{
				Attr result = SNG == thresh_type_ ? AttrSNGTrain(i, cur_info) : AttrDBLTrain(i, cur_info);	// 得到第i个属性作分类属性的结果
				if(result.max_gainratio_ > maxratio_result.max_gainratio_)
				{
					cur_attr_ = i;
					maxratio_result = result;			
				}
			}
		}
		
		// 如果属性用完或信息增益过小,停止生成子树
		if(cur_attr_ >= 0/* && maxratio_result.max_gainratio > 0.01*/)
		{
			// 输出分类信息
			if(isprint)
			{
				cout << '\t' << attr_array_.at(cur_attr_) << ':' << endl;
				PrintClassInfo(maxratio_result, type_);			
			}
			// 根据最大信息增益率的属性确定当前属性分类的阈值，并构造子树
			threshold_ = maxratio_result.threshold_;
			// 生成子树所需要的个案向量
			vector<Item> items1, items2;

			maxratio_result.inside_user_count_.clear();	// 这里的计数器需要重新计算，偶尔会和之前的有一两个偏差
			maxratio_result.outside_user_count_.clear();
			for(vector<Item>::const_iterator iter = items_.begin(); iter != items_.end(); ++iter)
			{
				if(SNG == thresh_type_ && iter->value_[cur_attr_] > maxratio_result.threshold_[0]
				|| DBL == thresh_type_ && (iter->value_[cur_attr_] < maxratio_result.threshold_[0]|| iter->value_[cur_attr_] > maxratio_result.threshold_[1]))
				{
					items1.push_back(*iter);
					++maxratio_result.outside_user_count_[iter->real_type_];
				}
				else
				{
					items2.push_back(*iter);
					++maxratio_result.inside_user_count_[iter->real_type_];
				}			
			}

			// items1代表两边(双阈值)或右边(单阈值)的类
			
			// 确定当前分类规则
			stringstream ss;
			if(SNG == thresh_type_)
			{
				ss << "(\"" << attr_array_[cur_attr_] << "\" > " << threshold_[0] << ')';
			}
			else
			{
				ss << "(\"" << attr_array_[cur_attr_] << "\" < " << threshold_[0]
					<< " || \"" << attr_array_[cur_attr_]  << "\" > " << threshold_[1] << ')';
			}
			string rule = ss.str();
				
			subTree_.push_back(
				DecisionTree(items1,
							maxratio_result.outside_user_count_,
							attr_array_,
							avai_attr_,
							used_attr_, 
							cur_attr_,
							positive_type_,
							negative_type_,
							thresh_type_,
							min_typenumber_,
							rule,
							depth_,
							max_depth_)
						);
			
			// items2代表中间(双阈值)或左边(单阈值)的类
			
			// 确定当前分类规则
			//ss.clear();
			ss.str("");	// stringstream对象的清空不能用clear()，clear()只是清除流状态标志(如出错标志)
			if(SNG == thresh_type_)
			{
				ss << "(\"" << attr_array_[cur_attr_] << "\" <= " << threshold_[0] << ')';
			}
			else
			{
				ss << "(\"" << attr_array_[cur_attr_] << "\" >= " << threshold_[0]
					<< " && \"" << attr_array_[cur_attr_]  << "\" <= " << threshold_[1] << ')';
			}
			rule.clear();
			rule = ss.str();

			subTree_.push_back(
				DecisionTree(items2,
							maxratio_result.inside_user_count_,
							attr_array_, 
							avai_attr_,
							used_attr_, 
							cur_attr_,
							positive_type_,
							negative_type_,
							thresh_type_,
							min_typenumber_,
							rule,
							depth_,
							max_depth_)
						);
		
			
			// 子树进行递归训练
			for(vector<DecisionTree>::iterator iter = subTree_.begin(); iter != subTree_.end(); ++iter)
			{
				iter->Train(isprint);
			}
		}
		else	// 不值得再分类时，令当前结点为叶子结点，并确定所表示的类别
		{			
			leaf_prob_ = static_cast<double>(positive) / static_cast<double>(positive + negative);	// 选比例较大者
//			cout << "positive: " << positive << '\t' << "negative: " << negative << endl;
		}
	}
	else if(type_.size() >= 1)	// 当只有一个类别或样本案例较少的时候
	{
		leaf_prob_ = static_cast<double>(positive) / static_cast<double>(positive + negative);	// 选比例较大者
//		cout << "positive: " << positive << '\t' << "negative: " << negative << endl;
	}
	else
	{
		cerr << "Error! Empty Type!" << endl;
	}

	items_ = temp_item;		// 恢复item的排列
}
void DecisionTree::LoadItem(const vector<Item> &items, bool IsLabeled)
{
	islabeled_ = IsLabeled;
	if(items.size() > 0)
	{
		items_ = items;
		cout << items.size() << " Item load successfully!" << endl;
		InitialTypeCount();
		// 确定正常类型
		map<string, int>::iterator iter = type_.begin();
		if(positive_type_ == iter->first)
		{
			++iter;	
		}
		negative_type_ = iter->first;
	}
	else
	{
		cerr << "items's size must be positive" << endl;
	}
}
void DecisionTree::LoadItem(const char *filename, bool IsLabeled)
{
	islabeled_ = IsLabeled;
	if(ReadSample(filename, IsLabeled) == false)
	{
		cerr << "Cannot open the file!" << endl;
		exit(1);
	}
	else
	{
		InitialTypeCount();
		// 确定正常类型
		map<string, int>::iterator iter = type_.begin();
		if(positive_type_ == iter->first)
		{
			++iter;	
		}
		negative_type_ = iter->first;
	}
}

double F_measure(const DecisionTree::ConfusionMatrix &m)
{
	double tp = static_cast<double>(m.true_positive_);
	double fp = static_cast<double>(m.false_positive_);
	double fn = static_cast<double>(m.false_negative_);
	double P = tp / (tp + fp);
	double R = tp / (tp + fn);
	return 2 * P * R / (P + R);
}

DecisionTree::ConfusionMatrix DecisionTree::Detect(bool isprint, double prob_thresh)
{
	// 对每个样本进行预测，并统计结果
	int positive(0), negative(0);
	int true_positive(0), false_positive(0);
	int true_negative(0), false_negative(0);
	for(vector<Item>::iterator iter = items_.begin(); iter != items_.end(); ++iter)
	{
		double positive_prob = Classify(*iter);
		if(positive_prob >= prob_thresh)
		{
			iter->predicted_type_ = positive_type_;
			++positive;
			if(true == islabeled_)
			{
				if(positive_type_ == iter->real_type_)
				{
					++true_positive;
				}
				else
				{
					++false_positive;
				}
			}
		}
		else	// if(negative_type_ == iter->predicted_type_)
		{
			iter->predicted_type_ = negative_type_;
			++negative;
			if(true == islabeled_)
			{
				if(negative_type_ == iter->real_type_)
				{
					++true_negative;
				}
				else
				{
					++false_negative;
				}
			}
		}
	}
	if(isprint)
	{
		if(true == islabeled_)
		{
			cout << "\t\t\tactual value" << endl;
			cout << "\t\t\tp (" << positive_type_ << ")\tn (" << negative_type_ << ")" << endl;
			cout << "prediction:\tp (" << positive_type_ << ")\t" << true_positive << "\t" << false_positive << endl;
			cout << "\t\tn (" << negative_type_ << ")\t" << false_negative << "\t" << true_negative << endl;
			cout << endl << endl;
			
			double precision = 0.0;
			double recall = 0.0;
			double devider = 0.0;
			if(true_positive > 0)
			{
				precision = static_cast<double>(true_positive) / positive;
				recall = static_cast<double>(true_positive) / (true_positive + false_negative);
			}
			cout << "precision:\t" << precision << endl;
			cout << "recall:\t\t" << recall  << endl;
			cout << "F-score:\t" << F_measure(ConfusionMatrix(true_positive, true_negative, false_positive, false_negative)) << endl;
		}
		else
		{
			cout << "positive:(" << positive_type_ << ")\t" << positive << endl;
			cout << "negative:(" << negative_type_ << ")\t" << negative << endl;
			true_positive = positive;
			true_negative = negative;
			false_positive = 0;
			false_negative = 0;
		}
	}

	return DecisionTree::ConfusionMatrix(true_positive, true_negative, false_positive, false_negative);
}



void DecisionTree::CrossValidation(size_t vali_times, double prob_thresh)	// 参数的意义为叶子结点中正例的比例达到多少才判定为正例
{
	if(false == islabeled_)
	{
		cerr << "cannot make validation without label!" << endl;
		exit(EXIT_FAILURE);
	}
	size_t ave_num = items_.size() / vali_times;
	vector<Item> train, train_front, train_back, test;
	train_front = items_;

	double P_sum(0.0), R_sum(0.0), F_sum(0.0); 
	ConfusionMatrix m_sum;
	for(size_t i = 0; i < vali_times; ++i)
	{
		if(test.size() > 0)					// 将test给train_back
		{
			for(size_t j = 0; j < ave_num; ++j)
			{			
				train_back.push_back(test.back());
				test.pop_back();
			}
		}

		test.clear();						// 将train_front的最后一份给test
		for(size_t j = 0; j < ave_num; ++j)
		{			
			test.push_back(train_front.back());
			train_front.pop_back();
		}

		train = train_front;				// 将train_front和train_back合起来
		vector<Item> temp = train_back;
		if(temp.size() > 0)
		{
			while(temp.size() > 0)
			{
				train.push_back(temp.back());
				temp.pop_back();
			}
		}

		LoadItem(train, true);
		Train(false);
		PrintRule();
		
		LoadItem(test, true);
		ConfusionMatrix m = Detect(true, prob_thresh);
		m_sum += m;
//		system("pause");

		double tp = static_cast<double>(m.true_positive_);
		double fp = static_cast<double>(m.false_positive_);
		double fn = static_cast<double>(m.false_negative_);
		
		double divider = tp + fp;
		if(divider < 1.0)
		{
			divider = 1.0;
		}
		double P = tp / divider;
		
		divider = tp + fn;
		if(divider < 1.0)
		{
			divider = 1.0;
		}
		double R = tp / divider;

		divider = P + R;
		if(divider < 1.0)
		{
			divider = 1.0;
		}
		double F = 2 * P * R / divider;
		P_sum += P;
		R_sum += R;
		F_sum += F;
	}
	cout << "average result:\n";
	cout << m_sum << endl;
	cout << "------------------------------------------------------" << endl;
	cout << "average precision:\t" << P_sum / static_cast<double>(vali_times) << endl;
	cout << "average recall:\t\t" << R_sum / static_cast<double>(vali_times) << endl;
	cout << "average F-score:\t" << F_sum / static_cast<double>(vali_times) << endl;
}

int DEPTH(0);
void DecisionTree::PrintRule() const
{
	if(rule_.size() > 0)
	{
		for(int i = 0; i < DEPTH; ++i)
		{
			cout << "    ";
		}
		cout << rule_ << endl;
	}
	if(subTree_.size() == 0)
	{
		for(int i = 0; i < DEPTH; ++i)
		{
			cout << "    ";
		}
		cout << "\tpositive: " << (type_.count(positive_type_) > 0 ? type_.at(positive_type_) : 0) << endl;
		for(int i = 0; i < DEPTH; ++i)
		{
			cout << "    ";
		}
		cout << "\tnegative: " << (type_.count(negative_type_) > 0 ? type_.at(negative_type_) : 0) << endl;
	}
	++DEPTH;
	for(vector<DecisionTree>::const_iterator iter = subTree_.begin(); iter != subTree_.end(); ++iter)
	{
//		cout << " && ";
		iter->PrintRule();
	}
	--DEPTH;
}
void DecisionTree::Print() const
{
	ostream& operator << (ostream &out, const Item &item);

	for(vector<Item>::const_iterator iter = items_.begin(); iter != items_.end(); ++iter)
	{
		cout << *iter << endl;
	}
}
void DecisionTree::Print(const char *filename) const
{
	ostream& operator << (ostream &out, const Item &item);

	ofstream out(filename);
	cout << filename << endl;
	if(!out)
	{
		cerr << "File Open ERROR in DecisionTree Print!" << endl;
		exit(1);
	}
	for(vector<Item>::const_iterator iter = items_.begin(); iter != items_.end(); ++iter)
	{
		out << *iter << endl;
	}
}
//DecisionTree& DecisionTree::operator = (const DecisionTree &root)
//{
//	attr_array_ = root.attr_array_;
//
//	return *this;
//}


/////////////////////////////////////
// Private Function

void DecisionTree::InitialTree()
{

}

bool DecisionTree::ReadSample(const char *filename, bool IsLabeled)
{
	istringstream& operator >> (istringstream &in, Item &item);

	if(NULL == filename)
	{
		return false;
	}

	ifstream in(filename);
	if(!in)
	{
		return false;
	}

	items_.clear();

	int count(0);
	string s;
	Item temp(attr_array_.size(), IsLabeled);
	istringstream s_in;
	cout << "Loading..." << endl;
	while(!in.eof())
	{
		getline(in, s);
		if(s.size() != 0)
		{
			s_in.str(s);
			s_in >> temp;
			items_.push_back(temp);
			++count;
		}
		s_in.clear();
	}
	cout << "Load " << count << " Item successfully!" << endl << endl;
	return true;
}

// 用载入的items_数组的元素来初始化type_
void DecisionTree::InitialTypeCount()
{
	type_.clear();
	for(vector<Item>::const_iterator iter = items_.begin(); iter != items_.end(); ++iter)
	{
		++type_[iter->real_type_];
	}
}

// 根据下标为attr_index的属性对实例items_进行排序
void DecisionTree::ItemSort(int attr_index)
{
	// 选择排序法对items_以value_的第attr_index个值进行顺序排序
	vector<Item>::iterator i, j, k;
	for(i = items_.begin(); i != items_.end(); ++i)
	{
		k = i;
		for(j = k + 1; j != items_.end(); ++j)
		{
			if(j->value_.at(attr_index) < k->value_.at(attr_index))
			{
				k = j;
			}
		}
		Item temp = *i;
		*i = *k;
		*k = temp;
	}
}
// 根据用户ID恢复实例items_的初始顺序
void DecisionTree::ItemInOrder()
{
	// 选择排序法
	vector<Item>::iterator i, j, k;
	for(i = items_.begin(); i != items_.end(); ++i)
	{
		k = i;
		for(j = k + 1; j != items_.end(); ++j)
		{
			if(j->id_ < k->id_)
			{
				k = j;
			}
		}
		Item temp = *i;
		*i = *k;
		*k = temp;
	}
}

// 清除样本
void DecisionTree::ClearItem()
{
	items_.clear();
	for(vector<DecisionTree>::iterator iter = subTree_.begin(); iter != subTree_.end(); ++iter)
	{
		iter->ClearItem();
	}
}

double DecisionTree::Info(map<string, int>type, map<string, int>::size_type size)
{
	double info(0.0);
	if(size < 0)
	{
		cerr << "Wrong! size cannot be negative!" << endl;
		exit(1);
	}
	else if(0 == size)
	{
		info = 0.0;
	}
	else
	{
		for(map<string, int>::const_iterator iter = type.begin(); iter != type.end(); ++iter)
		{
			info -= static_cast<double>(iter->second) / static_cast<double>(size) 
				* log(static_cast<double>(iter->second) / static_cast<double>(size));
		}
	}
	return info;
}
// AttrTrain函数，对attr_index所代表的的属性进行分类，函数中会更改subTree_和cur_attr_
DecisionTree::Attr DecisionTree::AttrSNGTrain(int attr_index, double parent_info)
{
	// 先按对应的属性将样本排序
	ItemSort(attr_index);

	double new_info(0.0), gain_info(0.0), split_info(0.0), gainratio(0.0), max_gainratio(0.0);	// 错误率
	// 双阈值将区间分成中段和边段，分别为(begin, l_threshold),[l_threshold, r_threshold],(r_threshold, end)
	// l_threshold的取值为[begin, end], r_threshold取值为[l_threshold, end]	(end表示最后一个元素而不是下一位)
	double threshold(0.0);
	double maxgr_threshold(0.0);

	map<string, int> left_type, right_type(type_);		// 左右两部分的样本频数计数器，左部分暂为空(无样本)，右部分暂同总计数器
	map<string, int> maxgr_left_type, maxgr_right_type;	// 记录信息增益率最大时的种类频数计数器
	vector<Item>::size_type left_size(0), right_size(items_.size());
	// 用迭代器iter表示items_的左边部分的末位元素,iter的取值为从items_的第一个位置直到倒数第二个位置
	double lpre_value(0.0);	// 这两个变量用于修复误差，用来处理相邻两个元素的属性值相同的情况
	vector<Item>::const_iterator post_iter;
	for(vector<Item>::const_iterator iter = items_.begin(), post_iter = iter; 
		iter != items_.end();
		++iter)
	{
		if(post_iter != items_.end())
		{
			++post_iter;	// r_iter的下一个位置的迭代器
		}
		if(post_iter != items_.end() 
			&& iter->value_[attr_index] == post_iter->value_[attr_index])
		{				
			++left_type[iter->real_type_];
			--right_type[iter->real_type_];
			++left_size;
			--right_size;
			continue;
		}

		// 左阈值更新
		threshold = iter->value_[attr_index];	
		// 新的信息熵			
		new_info = static_cast<double>(left_size) / static_cast<double>(items_.size()) * Info(left_type, left_size) 
				+ static_cast<double>(right_size) / static_cast<double>(items_.size()) * Info(right_type, right_size);
		gain_info = parent_info - new_info;
		split_info = 1.0;
		gainratio = gain_info / split_info;
		// 记下最大的信息增益率
		if(gainratio > max_gainratio)
		{
			max_gainratio = gainratio;
			maxgr_threshold = threshold;
			maxgr_left_type = left_type;
			maxgr_right_type = right_type;
		}		
	}

	// 返回一个存储属性节点信息的内嵌类
	return DecisionTree::Attr(items_.front().value_.at(attr_index), items_.back().value_.at(attr_index), // 该属性的min和max值
									vector<double>(1, maxgr_threshold),
									max_gainratio,		// 最小错误的阈值和最小错误率
									maxgr_left_type, maxgr_right_type);	// 错误率最小时的种类频数计数器
}
// AttrTrain函数，对attr_index所代表的的属性进行分类，函数中会更改subTree_和cur_attr_
DecisionTree::Attr DecisionTree::AttrDBLTrain(int attr_index, double parent_info)
{
	// 先按对应的属性将样本排序
	ItemSort(attr_index);

	double new_info(0.0), gain_info(0.0), split_info(0.0), gainratio(0.0), max_gainratio(0.0);	// 错误率
	// 双阈值将区间分成中段和边段，分别为(begin, l_threshold),[l_threshold, r_threshold],(r_threshold, end)
	// l_threshold的取值为[begin, end], r_threshold取值为[l_threshold, end]	(end表示最后一个元素而不是下一位)
	double l_threshold(0.0), r_threshold(0.0);
	double maxgr_lthreshold(0.0), maxgr_rthreshold(0.0);

	map<string, int> inside_type, outside_type(type_);		// 左右两部分的样本频数计数器，左部分暂为空(无样本)，右部分暂同总计数器
	map<string, int> maxgr_inside_type, maxgr_outside_type;	// 记录信息增益率最大时的种类频数计数器
	int inside_size(0), outside_size(0);
	// 用迭代器iter表示items_的左边部分的末位元素,iter的取值为从items_的第一个位置直到倒数第二个位置
//	string lpre_type, rpre_type;				// 用于简化迭代过程——但不起作用啊！！
	double lpre_value(0.0);	// 这两个变量用于修复误差，用来处理相邻两个元素的属性值相同的情况
	vector<Item>::const_iterator rpost_iter;
	for(vector<Item>::const_iterator l_iter = items_.begin(); l_iter != items_.end(); ++l_iter)
	{
		if(l_iter == items_.begin())
		{
//			lpre_type = l_iter->real_type_;
			lpre_value = l_iter->value_[attr_index];
		}
		else if(l_iter->value_[attr_index] == lpre_value)
		{
			continue;
		}
		// 当前值的记录(用于修复下次迭代的误差和简化迭代次数)
//		lpre_type = l_iter->real_type_;
		lpre_value = l_iter->value_[attr_index];
		// 计数器初始化
		inside_type.clear();						
		outside_type = type_;
		// 长度初始化
		inside_size = 0;
		outside_size = items_.size();
		// 左阈值更新
		l_threshold = l_iter->value_[attr_index];	
		for(vector<Item>::const_iterator r_iter = l_iter, rpost_iter = r_iter;
			r_iter != items_.end();
			++r_iter)
		{
			if(rpost_iter != items_.end())
			{
				++rpost_iter;	// r_iter的下一个位置的迭代器
			}
			if(rpost_iter != items_.end() 
				&& r_iter->value_[attr_index] == rpost_iter->value_[attr_index])
			{				
				++inside_type[r_iter->real_type_];
				--outside_type[r_iter->real_type_];
				++inside_size;
				--outside_size;
				continue;
			}
			
			// 当前值的记录(用于修复下次迭代的误差和简化迭代次数)
//			rpre_type = r_iter->real_type_;

			r_threshold = r_iter->value_[attr_index];	// 右阈值更新
			++inside_type[r_iter->real_type_];	
			--outside_type[r_iter->real_type_];
			++inside_size;
			--outside_size;
			
			new_info = static_cast<double>(inside_size) / static_cast<double>(items_.size()) * Info(inside_type, inside_size) 
				 + static_cast<double>(outside_size) / static_cast<double>(items_.size()) * Info(outside_type, outside_size);
			gain_info = parent_info - new_info;
			split_info = 1;
			gainratio = gain_info / split_info;
			// 记下最大的信息增益率
			if(gainratio > max_gainratio)
			{
				max_gainratio = gainratio;
				maxgr_lthreshold = l_threshold;
				maxgr_rthreshold = r_threshold;
				maxgr_inside_type = inside_type;
				maxgr_outside_type = outside_type;
			}
		}
	}

	vector<double> threshold;
	threshold.push_back(maxgr_lthreshold);
	threshold.push_back(maxgr_rthreshold);
	// 返回一个存储属性节点信息的内嵌类
	return DecisionTree::Attr(items_.front().value_.at(attr_index), items_.back().value_.at(attr_index), // 该属性的min和max值
									threshold,
									max_gainratio,		// 最小错误的阈值和最小错误率
									maxgr_inside_type, maxgr_outside_type);	// 错误率最小时的种类频数计数器
}


double DecisionTree::Classify(const Item &item)
{
	if(subTree_.size() == 0)
	{
		return leaf_prob_;
	}
	else
	{
		if(SNG == thresh_type_ && item.value_[cur_attr_] > threshold_[0]
			|| DBL == thresh_type_ && (item.value_[cur_attr_] < threshold_[0] || item.value_[cur_attr_] > threshold_[1]))
		{
			return subTree_[0].Classify(item);	// 第一个子树是两边或右边的区间
		}
		else
		{
			return subTree_[1].Classify(item);	// 第二个子树是中间或左边的区间
		}
	}
}

ostream& operator << (ostream &out, const DecisionTree::ConfusionMatrix &c_mat)
{
	out << "\t\t\tactual value" << '\n';
	out << "\t\t\tp (1)\tn (-1)" << '\n';
	out << "prediction:\tp (1)\t" << c_mat.true_positive_ << "\t" << c_mat.false_positive_ << '\n';
	out << "\t\tn (-1)\t" << c_mat.false_negative_ << "\t" << c_mat.true_negative_ << '\n';
	out << '\n';
			
	double precision = 0.0;
	double accuracy = 0.0;
	double recall = 0.0;
	double devider = 0.0;
	int total_sample = c_mat.true_positive_ + c_mat.true_negative_ + c_mat.false_positive_ + c_mat.false_negative_;

	if(c_mat.true_positive_ + c_mat.false_positive_ > 0)
	{
		precision = static_cast<double>(c_mat.true_positive_) / static_cast<double>(c_mat.true_positive_ + c_mat.false_positive_);
	}
	else
	{
		precision = 1.0;
	}
	
	if(total_sample > 0)
	{
		accuracy = static_cast<double>(c_mat.true_positive_ + c_mat.true_negative_) /
			static_cast<double>(c_mat.true_positive_ + c_mat.true_negative_ + c_mat.false_positive_ + c_mat.false_negative_);
	}
	else
	{
		accuracy = 1.0;
	}

	if(c_mat.true_positive_ + c_mat.false_negative_ > 0)
	{
		recall = static_cast<double>(c_mat.true_positive_) / static_cast<double>(c_mat.true_positive_ + c_mat.false_negative_);
	}
	else
	{
		recall = 1.0;
	}

	double f1_score = 2 * precision * recall / (precision + recall);
	out << "precision:\t" << precision << '\n';
	out << "accuracy:\t" << accuracy << '\n';
	out << "recall:\t\t" << recall  << '\n';
	out << "F-score:\t" << f1_score << '\n';
	return out;
}