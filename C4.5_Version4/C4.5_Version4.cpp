// C4.5_Version4.cpp : �������̨Ӧ�ó������ڵ㡣
//

#include "stdafx.h"


int _tmain(int argc, _TCHAR* argv[])
{
	const char *filename = "bi_labeled_feature4101.txt";
	string attr_array[] = {"��ע","��˿","����","��ע��˿��","��ע���۱�","�û����Ƹ��Ӷ�","΢������","�¾�΢��","΢������ʱ����",
		"ת������","���ӱ���","ƽ��������", "ԭ��΢��������","΢��ƽ������","�������ƶ�","ģ���ƶ�","��������ƶ�"};
	bool avai_attr[] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
//	bool avai_attr[17] = {1,1};
	// ��һ������Ϊ����������
	// �ڶ�����������Ҫ��ȡ�����ı�ǩ
	int feature_num = sizeof(avai_attr) / sizeof(avai_attr[0]);
	DecisionTree root(vector<string>(attr_array, attr_array + feature_num), vector<bool>(avai_attr, avai_attr + feature_num),
		"-1", DecisionTree::SNG,
		10, 4);	// ����һ��DecisionTree���󣬵���filename�ļ��е�ʵ�����������С������Ϊ10������������Ϊ4
//	system("pause");
	root.LoadItem(filename, true);
//	system("pause");
	root.Train(true);		// ��������ѵ����������
	cout << "ѵ����ɣ�����������������";
	system("pause > tmp");
	cout << endl	<< endl << "Decision Tree:" << endl << endl;
	root.PrintRule();
	cout << "���������������" << endl;
	system("pause > tmp");
	root.Detect(true, 0.5);	// �Լ��
	cout << "����������5�����������" << endl;
	system("pause > tmp");
	/*cout << "FileFoldValidation!\n";
	*/root.CrossValidation(5, 0.45);
	return 0;
}

