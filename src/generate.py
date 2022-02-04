import gc
import csv
import math
from datetime import datetime
from collections import defaultdict

def func_mean(lst):
    return sum(lst) / float(len(lst))
    
def func_median(lst):
    even = (0 if len(lst) % 2 else 1) + 1
    half = (len(lst) - 1) // 2
    return sum(sorted(lst)[half:half + even]) / float(even)
    
def func_reduce_data(path_to_offers, path_to_transactions, path_to_reduced):
    start = datetime.now()
    offers_category = {}
    offers_dept = {}
    offers_company = {}
    offers_brand = {}
    for e, line in enumerate( open(path_to_offers) ):
        category = line.split(',')[1]
        if e == 0:
            dept = 'dept'
        else:
            dept = str(int(math.floor(float(category)/100.0)))
        company = line.split(',')[3]
        brand = line.split(',')[5]
        offers_company[ company ] = 1
        offers_category[ category ] = 1
        offers_dept[ dept ] = 1
        offers_brand[ brand ] = 1
    with open(path_to_reduced, 'w') as outfile:
        reduced = 0
        for e, line in enumerate( open(path_to_transactions) ):
            if e == 0:
                outfile.write( line )
            else:
                category = line.split(',')[3]
                dept = str(int(math.floor(float(category)/100.0)))
                company = line.split(',')[4]
                brand = line.split(',')[5]
                if dept in offers_dept or company in offers_company or brand in offers_brand:
                    outfile.write( line )
                    reduced += 1


def func_diff_days(s1,s2):
    format = '%Y-%m-%d'
    return (datetime.strptime(s2, format) - datetime.strptime(s1, format)).days


def func_generate_features(path_to_train, path_to_test, path_to_offers, path_to_transactions, date_diff_days_thresh, path_to_out_train, path_to_out_test):
    offers = {}
    with open( path_to_offers ) as offer_file:
        for e, line in enumerate( offer_file ):
            row = line.strip().split(',')
            offers[ row[0] ] = row
    
    train_history = {}
    with open( path_to_train ) as train_history_file:
        for e, line in enumerate( train_history_file ):
            if e > 0:
                row = line.strip().split(',')
                train_history[ row[0] ] = row
    test_history = {}
    with open( path_to_test ) as test_history_file:
        for e, line in enumerate( test_history_file ):
            if e > 0:
                row = line.strip().split(',')
                test_history[ row[0] ] = row
    
    features_list_train = []
    features_list_test = []
    done_header_row_train = False
    done_header_row_test = False
    cache_row_size = 10000
    header_names = set()
    features_overall = defaultdict(float)
    
    path_to_out_train_shopper = path_to_out_train[:-4] + '_shopper.csv'
    path_to_out_test_shopper = path_to_out_test[:-4] + '_shopper.csv'
    out_train_shopper = open( path_to_out_train_shopper, 'wb' )
    out_test_shopper = open( path_to_out_test_shopper, 'wb' )
    with open( path_to_transactions ) as transactions_file:
        last_id = 0
        features = defaultdict(float)
        for e, line in enumerate( transactions_file ):
            if e > 500000:
                break
            elif e > 0:
                row = line.strip().split(',')
                
                row_id = row[0]
                row_company = row[4]
                row_category = row[3]
                row_brand = row[5]
                row_dept = math.floor(float(row_category)/100.0)
                row_date = row[6]
                
                size = float( row[7] )
                quantity = float( row[9] )
                amount = float( row[10] )
                TYPES_VALUES = { 't': 1.0, 's': size, 'q': quantity, 'a': amount }
                
                KEYS = ['overall_bought_company{0}'.format(row_company),
                        'overall_bought_category{0}'.format(row_category),
                        'overall_bought_brand{0}'.format(row_brand),
                        'overall_bought_dept{0}'.format(row_dept),
                        'overall_bought_company{0}_category{1}'.format(row_company,row_category),
                        'overall_bought_company{0}_brand{1}'.format(row_company,row_brand),
                        'overall_bought_category{0}_brand{1}'.format(row_category,row_brand),
                        'overall_bought_company{0}_dept{1}'.format(row_company,row_dept),
                        'overall_bought_dept{0}_brand{1}'.format(row_dept,row_brand),
                        'overall_bought_company{0}_category{1}_brand{2}'.format(row_company,row_category,row_brand),
                        'overall_bought_company{0}_dept{1}_brand{2}'.format(row_company,row_dept,row_brand)
                        ]
                        
                for k in KEYS:
                    for tt, vv in TYPES_VALUES.items():
                        features_overall['{0}_{1}'.format(k,tt)] += vv
                    
                if last_id != row_id and e != 1:
                    
                    D = {
                          'shopper_never_bought_offer_company': 'shopper_bought_offer_company_t' in features,
                          'shopper_never_bought_offer_category': 'shopper_bought_offer_category_t' in features,
                          'shopper_never_bought_offer_brand': 'shopper_bought_offer_brand_t' in features,
                          'shopper_never_bought_offer_dept': 'shopper_bought_offer_dept_t' in features,
                          'shopper_never_bought_offer_company_category': 'shopper_bought_offer_company_category_t' in features,
                          'shopper_never_bought_offer_company_brand': 'shopper_bought_offer_company_brand_t' in features,
                          'shopper_never_bought_offer_category_brand': 'shopper_bought_offer_category_brand_t' in features,
                          'shopper_never_bought_offer_company_dept': 'shopper_bought_offer_company_dept_t' in features,
                          'shopper_never_bought_offer_dept_brand': 'shopper_bought_offer_dept_brand_t' in features,
                          'shopper_never_bought_offer_company_category_brand': 'shopper_bought_offer_company_category_brand_t' in features,
                          'shopper_never_bought_offer_company_dept_brand': 'shopper_bought_offer_company_dept_brand_t' in features
                    }
                    for k, v in D.items():
                        if not v:
                            features[k] = 1
                        else:
                            features[k] = 0
                            
                    features['shopper_bought_size_median'] = func_median( features['shopper_bought_size_median'] )
                    features['shopper_bought_quantity_median'] = func_median( features['shopper_bought_quantity_median'] )
                    features['shopper_bought_amount_median'] = func_median( features['shopper_bought_amount_median'] )
                    
                    this_company = float(features['shopper_bought_company_num'][features['company']])
                    all_company = float(sum(features['shopper_bought_company_num'].values()))
                    features['shopper_bought_offer_company_ratio'] = this_company / all_company
                    features['shopper_bought_company_median_time'] = func_median(features['shopper_bought_company_num'].values())
                    features['shopper_bought_company_num'] = len(features['shopper_bought_company_num'].keys())
                    
                    this_category = float(features['shopper_bought_category_num'][features['category']])
                    all_category = float(sum(features['shopper_bought_category_num'].values()))
                    features['shopper_bought_offer_category_ratio'] = this_category / all_category
                    features['shopper_bought_category_median_time'] = func_median(features['shopper_bought_category_num'].values())
                    features['shopper_bought_category_num'] = len(features['shopper_bought_category_num'].keys())
                    
                    this_dept = float(features['shopper_bought_dept_num'][features['dept']])
                    all_dept = float(sum(features['shopper_bought_dept_num'].values()))
                    features['shopper_bought_offer_dept_ratio'] = this_dept / all_dept
                    features['shopper_bought_dept_median_time'] = func_median(features['shopper_bought_dept_num'].values())
                    features['shopper_bought_dept_num'] = len(features['shopper_bought_dept_num'].keys())
                    
                    this_brand = float(features['shopper_bought_brand_num'][features['brand']])
                    all_brand = float(sum(features['shopper_bought_brand_num'].values()))
                    features['shopper_bought_offer_brand_ratio'] = this_brand / all_brand
                    features['shopper_bought_brand_median_time'] = func_median(features['shopper_bought_brand_num'].values())
                    features['shopper_bought_brand_num'] = len(features['shopper_bought_brand_num'].keys())
                    
                    features['shopper_bought_date_median_time'] = func_median(features['shopper_bought_date_num'].values())
                    date_ = features['shopper_bought_date_num'].keys()
                    features['shopper_bought_date_num'] = len(date_)

                    if len(date_) == 1:
                        date_gap = [0]
                    else:
                        date_gap = []
                        date_ = sorted(date_)
                        for i in range(len(date_)-1):
                            date_gap.append( func_diff_days(date_[i], date_[i+1]) )
                    features['shopper_bought_date_median_gap'] = func_median(date_gap)
                    
                    if features['repeater'] == 0.5:
                        if done_header_row_train == False:
                            features_list_test.append( features )
                            header_names.update( features.keys() )
                        else:
                            if done_header_row_test == False:
                                features_list_test.append( features )
                                features_list_test.insert(0, header_row)
                                writer_test = csv.DictWriter(out_test_shopper, header_names)
                                writer_test.writerows( features_list_test )
                                done_header_row_test = True
                                del features_list_test
                                gc.collect()
                            else:
                                writer_test.writerow( features )
                    else:
                        if done_header_row_train == False:
                            features_list_train.append( features )
                            header_names.update( features.keys() )
                            if len(features_list_train) == cache_row_size:
                                header_names = list(header_names)
                                key_to_adjust = ['id', 'offer', 'repeater', 'repeattrips', 'offer_value',
                                                 'offer_date', 'offer_days', 'offer_mday', 'offer_mweek', 'offer_weekday', 
                                                 'company', 'category', 'dept', 'brand', 'market', 'chain' ]
                                for k in key_to_adjust:
                                    header_names.remove(k)
                                for k in key_to_adjust[::-1]:
                                    header_names.insert(0, k)
                                header_row = {}
                                for key in header_names:
                                    header_row[key] = key
                                features_list_train.insert(0, header_row)
                                writer_train = csv.DictWriter(out_train_shopper, header_names)
                                writer_train.writerows( features_list_train )
                                done_header_row_train = True
                                del features_list_train
                                gc.collect()
                        else:
                            writer_train.writerow( features )
                    
                    features = defaultdict(float)
                if row_id in train_history or row_id in test_history:
                    features['id'] = row_id
                    if row_id in train_history:
                        if train_history[row_id][5] == 't':
                            features['repeater'] = 1
                        else:
                            features['repeater'] = 0
                        features['repeattrips'] = train_history[row_id][4]                        
                        history = train_history[row_id]
                    else:
                        features['repeater'] = 0.5
                        features['repeattrips'] = 0.5
                        history = test_history[row_id]
                        
                    offer_company = offers[ history[2] ][3]
                    offer_category = offers[ history[2] ][1]
                    offer_brand = offers[ history[2] ][5]
                    offer_dept = math.floor(float(offer_category)/100.0)
                    
                    date_diff_days = func_diff_days(row_date, history[-1])
                    
                    features['company'] = offer_company
                    features['category'] = offer_category
                    features['brand'] = offer_brand
                    features['dept'] = offer_dept
                    
                    features['chain'] = history[1]
                    features['offer'] = history[2]
                    features['market'] = history[3]
                    features['offer_value'] = offers[ history[2] ][4]
                    offer_date = history[-1]
                    date_format = '%Y-%m-%d'
                    offer_date = datetime.strptime(offer_date, date_format)
                    features['offer_date'] = history[-1]
                    base_date = '2013-03-01'
                    base_date = datetime.strptime(base_date, date_format)
                    features['offer_days'] = (offer_date - base_date).days
                    features['offer_mday'] = offer_date.day
                    features['offer_mweek'] = math.floor(offer_date.day/4.0)

                    features['offer_weekday'] = offer_date.weekday()

                    
                    features['shopper_bought_time_total'] += 1.0                    
                    features['shopper_bought_size_total'] += size
                    features['shopper_bought_quantity_total'] += quantity
                    features['shopper_bought_amount_total'] += amount
                    
                    if not 'shopper_bought_size_median' in features:
                        features['shopper_bought_size_median'] = [ size ]
                    else:
                        features['shopper_bought_size_median'].append( size )
                        
                    if not 'shopper_bought_quantity_median' in features:
                        features['shopper_bought_quantity_median'] = [ quantity ]
                    else:
                        features['shopper_bought_quantity_median'].append( quantity )
                        
                    if not 'shopper_bought_amount_median' in features:
                        features['shopper_bought_amount_median'] = [ amount ]
                    else:
                        features['shopper_bought_amount_median'].append( amount )

                    if not 'shopper_bought_company_num' in features:
                        features['shopper_bought_company_num'] = defaultdict(float)
                    features['shopper_bought_company_num'][row_company] += 1.0
                        
                    if not 'shopper_bought_category_num' in features:
                        features['shopper_bought_category_num'] = defaultdict(float)
                    features['shopper_bought_category_num'][row_category] += 1.0
                    
                    if not 'shopper_bought_dept_num' in features:
                        features['shopper_bought_dept_num'] = defaultdict(float)
                    features['shopper_bought_dept_num'][row_dept] += 1.0
                    
                    if not 'shopper_bought_brand_num' in features:
                        features['shopper_bought_brand_num'] = defaultdict(float)
                    features['shopper_bought_brand_num'][row_brand] += 1.0

                    if not 'shopper_bought_date_num' in features:
                        features['shopper_bought_date_num'] = defaultdict(float)
                    features['shopper_bought_date_num'][row_date] += 1.0
                    
                    D = {
                          'shopper_returned': amount < 0 or size < 0 or quantity < 0,
                          'shopper_bought_offer_company': offer_company == row_company,
                          'shopper_bought_offer_category': offer_category == row_category,
                          'shopper_bought_offer_dept': offer_dept == row_dept,
                          'shopper_bought_offer_brand': offer_brand == row_brand,
                          'shopper_bought_offer_company_category': offer_company == row_company and offer_category == row_category,
                          'shopper_bought_offer_company_brand': offer_company == row_company and offer_brand == row_brand,
                          'shopper_bought_offer_category_brand': offer_category == row_category and offer_brand == row_brand,
                          'shopper_bought_offer_company_dept': offer_company == row_company and offer_dept == row_dept,
                          'shopper_bought_offer_dept_brand': offer_dept == row_dept and offer_brand == row_brand,
                          'shopper_bought_offer_company_category_brand': offer_company == row_company and offer_category == row_category and offer_brand == row_brand,
                          'shopper_bought_offer_company_dept_brand': offer_company == row_company and offer_dept == row_dept and offer_brand == row_brand
                    }

                    for k, v in D.items():
                        if v == 1:
                            for tt, vv in TYPES_VALUES.items():
                                if k == 'shopper_returned':
                                    vv = abs(vv)
                                features['{0}_{1}'.format(k,tt)] += vv
                            for thresh in date_diff_days_thresh:
                                if date_diff_days < thresh:
                                    for tt, vv in TYPES_VALUES.items():
                                        if k == 'shopper_returned':
                                            vv = abs(vv)
                                        features['{0}_{1}_{2}'.format(k,tt,thresh)] += vv
                                
                last_id = row_id
                                   
    out_train_shopper.close()
    out_test_shopper.close()
    K = [
          'overall_bought_company',
          'overall_bought_category',
          'overall_bought_brand',
          'overall_bought_dept',
          'overall_bought_company_category',
          'overall_bought_company_brand',
          'overall_bought_category_brand',
          'overall_bought_company_dept',
          'overall_bought_dept_brand',
          'overall_bought_company_category_brand',
          'overall_bought_company_dept_brand'
    ]
    KEYS = []
    for k in K:
        for tt in TYPES_VALUES.keys():
            KEYS.append( '{0}_{1}'.format(k, tt) )
    header_names = header_names + KEYS
    header_row = {}
    for key in header_names:
        header_row[key] = key
    
    out_train_shopper = open(path_to_out_train_shopper, 'rb')
    with open(path_to_out_train, 'wb') as out_train:
        writer_train = csv.DictWriter(out_train, header_names)
        writer_train.writerow( header_row )
        for e, line in enumerate( out_train_shopper ):
            if e > 0:
                f = line.split(',')
                features = {}
                for k, v in zip(header_names, f):
                    features[k] = v
                features = func_combine_shopper_overall_features(features, features_overall)
                writer_train.writerow( features )
    out_train_shopper.close()

    out_test_shopper = open(path_to_out_test_shopper, 'rb')
    with open(path_to_out_test, 'wb') as out_test:
        writer_test = csv.DictWriter(out_test, header_names)
        writer_test.writerow( header_row )
        for e, line in enumerate( out_test_shopper ):
            if e > 0:
                f = line.split(',')
                features = {}
                for k, v in zip(header_names, f):
                    features[k] = v
                features = func_combine_shopper_overall_features(features, features_overall)
                writer_test.writerow( features )
    out_test_shopper.close()
    
def func_combine_shopper_overall_features(features, features_overall):
    company = features['company']
    category = features['category']
    brand = features['brand']
    dept = features['dept']
    D = {
          'overall_bought_company': 'overall_bought_company{0}'.format(company),
          'overall_bought_category': 'overall_bought_category{0}'.format(category),
          'overall_bought_brand':'overall_bought_brand{0}'.format(brand),
          'overall_bought_dept':'overall_bought_dept{0}'.format(dept),
          'overall_bought_company_category':'overall_bought_company{0}_category{1}'.format(company,category),
          'overall_bought_company_brand':'overall_bought_company{0}_brand{1}'.format(company,brand),
          'overall_bought_category_brand':'overall_bought_category{0}_brand{1}'.format(category,brand),
          'overall_bought_company_dept':'overall_bought_company{0}_dept{1}'.format(company,dept),
          'overall_bought_dept_brand':'overall_bought_dept{0}_brand{1}'.format(dept,brand),
          'overall_bought_company_category_brand':'overall_bought_company{0}_category{1}_brand{2}'.format(company,category,brand),
          'overall_bought_company_dept_brand':'overall_bought_company{0}_dept{1}_brand{2}'.format(company,dept,brand)
    }
    TYPES = [ 't', 's', 'q', 'a' ]
    for k,v in D.items():
        for tt in TYPES:
            features['{0}_{1}'.format(k,tt)] = features_overall['{0}_{1}'.format(v,tt)]
            
    return features

if __name__ == '__main__':
    path_to_offers = '../Data/offers.csv'
    path_to_transactions = '../Data/transactions.csv'
    path_to_train = '../Data/trainHistory.csv'
    path_to_test = '../Data/testHistory.csv'

    path_to_reduced = '../Data/reduced.csv'
    path_to_out_train = '../Data/train.csv'
    path_to_out_test = '../Data/test.csv'

    date_diff_days_thresh = [ 10, 20, 30, 45, 60, 90, 120, 180, 240, 360 ]
    func_reduce_data(path_to_offers, path_to_transactions, path_to_reduced)
    func_generate_features(path_to_train, path_to_test, path_to_offers, path_to_transactions,
                            date_diff_days_thresh, path_to_out_train, path_to_out_test)
