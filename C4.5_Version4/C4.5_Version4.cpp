// C4.5_Version4.cpp : 定义控制台应用程序的入口点。
//

#include "stdafx.h"


int _tmain(int argc, _TCHAR* argv[])
{
	const char *filename = "bi_labeled_feature4101.txt";
	string attr_array[] = {"关注","粉丝","互粉","关注粉丝比","关注互粉比","用户名称复杂度","微博总数","月均微博","微博发布时间间隔",
		"转发比例","链接比例","平均评论数", "原创微博评论数","微博平均长度","余弦相似度","模相似度","共享词相似度"};
	bool avai_attr[] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
//	bool avai_attr[17] = {1,1};
	// 第一个参数为特征向量，
	// 第二个参数是需要提取的类别的标签
	int feature_num = sizeof(avai_attr) / sizeof(avai_attr[0]);
	DecisionTree root(vector<string>(attr_array, attr_array + feature_num), vector<bool>(avai_attr, avai_attr + feature_num),
		"-1", DecisionTree::SNG,
		10, 4);	// 定义一个DecisionTree对象，导入filename文件中的实例。分类的最小个案数为10，树的最大深度为4
//	system("pause");
	root.LoadItem(filename, true);
//	system("pause");
	root.Train(true);		// 根据样本训练出分类器
	cout << "训练完成，按任意键输出决策树";
	system("pause > tmp");
	cout << endl	<< endl << "Decision Tree:" << endl << endl;
	root.PrintRule();
	cout << "按任意键输出检测结果" << endl;
	system("pause > tmp");
	root.Detect(true, 0.5);	// 自检测
	cout << "按任意键输出5倍交叉检验结果" << endl;
	system("pause > tmp");
	/*cout << "FileFoldValidation!\n";
	*/root.CrossValidation(5, 0.45);
	return 0;
}

