#include "stdafx.h"

Item::Item(int attr_count, bool isLabeled) : value_(attr_count), isLabeled_(isLabeled)
{
	
}

ostream& operator << (ostream &out, const Item &item)
{
	out << item.id_ << '\t';
	for(vector<double>::const_iterator iter = item.value_.begin(); iter != item.value_.end(); ++iter)
	{
		out << *iter << '\t';
	}
	if(true == item.isLabeled_)
	{
		out << item.real_type_ << '\t';
	}
	out << item.predicted_type_ << endl;
	return out;
}

istringstream& operator >> (istringstream &in, Item &item)
{
	in >> item.id_;
	for(vector<double>::size_type i = 0; i < item.value_.size(); ++i)
	{
		in >> item.value_.at(i);
	}
	if(true == item.isLabeled_)
	{
		in >> item.real_type_;
		if (item.real_type_.compare("1") != 0
			&& item.real_type_.compare("-1") != 0)
		{
			cerr << "TAG VALUE ERROR!" << endl;
			throw std::runtime_error("TAG VALUE ERROR!");
		}
	}
	return in;
}