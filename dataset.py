import pickle
import numpy as np
import torch
from collections import defaultdict

def build_city_to_province_map(area_keys):
    # Mapping Chinese prefecture-level cities to provinces
    prov_map = {
        '浙江省': ['温州市', '宁波市', '舟山市', '杭州市', '衢州市', '金华市', '湖州市', '台州市', '绍兴市', '丽水市', '嘉兴市'],
        '江苏省': ['徐州市', '连云港市', '淮安市', '苏州市', '南京市', '宿迁市', '无锡市', '镇江市', '盐城市', '常州市', '南通市', '泰州市'],
        '江西省': ['上饶市', '宜春市', '九江市', '鹰潭市', '抚州市', '南昌市', '吉安市', '新余市', '萍乡市'],
        '福建省': ['泉州市', '宁德市', '南平市', '厦门市', '漳州市', '福州市', '三明市'],
        '山东省': ['青岛市', '潍坊市', '济宁市', '临沂市', '滨州市', '济南市', '菏泽市', '德州市', '东营市', '威海市', '枣庄市', '淄博市'],
        '广东省': ['深圳市', '东莞市', '广州市', '珠海市', '揭阳市', '佛山市', '江门市'],
        '安徽省': ['芜湖市', '阜阳市', '合肥市', '蚌埠市', '黄山市', '淮南市', '铜陵市', '马鞍山市', '宣城市', '安庆市'],
        '湖北省': ['荆门市', '襄阳市', '武汉市', '黄冈市', '宜昌市', '黄石市'],
        '湖南省': ['娄底市', '岳阳市', '邵阳市', '张家界市', '长沙市', '永州市', '衡阳市', '湘潭市'],
        '四川省': ['成都市', '绵阳市'],
        '河南省': ['许昌市', '驻马店市', '郑州市', '洛阳市', '安阳市', '商丘市', '开封市'],
        '河北省': ['邯郸市', '石家庄市', '秦皇岛市', '沧州市', '保定市', '廊坊市'],
        '山西省': ['忻州市', '太原市', '晋城市', '运城市'],
        '辽宁省': ['鞍山市', '大连市', '沈阳市', '朝阳市', '盘锦市', '铁岭市'],
        '吉林省': ['延边朝鲜族自治州', '长春市'],
        '黑龙江省': ['佳木斯市', '双鸭山市', '绥化市', '大庆市', '牡丹江市'],
        '内蒙古自治区': ['乌兰察布市', '呼和浩特市', '兴安盟', '锡林郭勒盟'],
        '陕西省': ['西安市', '汉中市', '铜川市'],
        '甘肃省': ['兰州市'],
        '青海省': ['海西蒙古族藏族自治州', '西宁市'],
        '宁夏回族自治区': ['银川市', '吴忠市'],
        '新疆维吾尔自治区': ['乌鲁木齐市', '伊犁哈萨克自治州', '石河子市'],
        '西藏自治区': ['拉萨市'],
        '云南省': ['昭通市', '昆明市', '大理白族自治州', '怒江傈僳族自治州'],
        '贵州省': ['贵阳市'],
        '广西壮族自治区': ['桂林市', '防城港市', '柳州市'],
        '海南省': ['东方市', '海口市', '澄迈县', '陵水黎族自治县'],
        '北京市': ['北京市'],
        '上海市': ['上海市'],
        '天津市': ['天津市'],
        '重庆市': ['重庆市'],
        '台湾省': ['台南市', '苗栗县', '金门县', '屏东县', '云林县', '新竹县'],
    }
    city_to_prov = {}
    for prov, cities in prov_map.items():
        for c in cities:
            city_to_prov[c] = prov
    
    unique_provs = sorted(list(prov_map.keys())) + ['其他']
    prov_to_idx = {p: i for i, p in enumerate(unique_provs)}
    
    city_prov_idx = []
    for a in area_keys:
        p = city_to_prov.get(str(a), '其他')
        city_prov_idx.append(prov_to_idx[p])
    return city_prov_idx, len(unique_provs)

def build_industry_to_sector_map(ind_keys):
    sector_keywords = [
        ('制造业', ['制造', '生产', '加工', '制品', '汽车', '机械', '电器', '印刷', '纺织', '服装', '钢铁', '设备', '仪器']),
        ('批发和零售业', ['批发', '零售']),
        ('建筑业', ['建筑', '土木', '工程', '安装', '装饰']),
        ('农林牧渔业', ['农', '林', '牧', '渔']),
        ('交通运输仓储业', ['运输', '仓储', '搬运', '货运', '铁路', '船舶', '航空']),
        ('信息和软件服务业', ['软件', '信息', '网络', '互联网', '计算机', '通信', '传媒']),
        ('科学研究和技术服务业', ['研究', '试验', '科技', '技术', '推广']),
        ('电力热力燃气水生产供应业', ['电力', '热力', '燃气', '水务', '水的生产']),
        ('金融业', ['金融', '资本', '银行', '证券', '保险', '货币']),
        ('房地产业', ['房地产', '房屋']),
        ('租赁和商务服务业', ['商务', '租赁']),
        ('水利环境公共设施管理业', ['水利', '环境', '公共设施', '生态']),
        ('采矿业', ['采选', '开采', '矿']),
        ('居民服务修理业', ['居民', '修理', '日用']),
        ('住宿和餐饮业', ['餐饮', '住宿', '旅游']),
        ('文化体育娱乐业', ['文化', '体育', '娱乐', '影视', '广播']),
        ('教育', ['教育']),
    ]
    sectors = [s[0] for s in sector_keywords] + ['其他']
    sec_to_idx = {s: i for i, s in enumerate(sectors)}
    
    ind_sec_idx = []
    for k in ind_keys:
        k_str = str(k)
        matched = False
        for sec_name, kws in sector_keywords:
            if any(kw in k_str for kw in kws):
                ind_sec_idx.append(sec_to_idx[sec_name])
                matched = True
                break
        if not matched:
            ind_sec_idx.append(sec_to_idx['其他'])
    return ind_sec_idx, len(sectors)

def load_bphd_data(data_dir=r'D:\SDI\ComRisk\data', threshold_t=0.05):
    # 1. Load splits
    with open(f'{data_dir}/split_data_idx.pkl', 'rb') as f:
        train_idx, valid_idx, test_idx = pickle.load(f)

    # Clean data leakage: 17 test companies in train_idx are removed to get the clean 474 test set
    train_set = set(train_idx)
    clean_test_mask = [idx not in train_set for idx in test_idx]
    clean_test_idx = [idx for idx in test_idx if idx not in train_set]
    clean_test_positions = [i for i, idx in enumerate(test_idx) if idx not in train_set]

    # Clean valid
    clean_valid_mask = [idx not in train_set for idx in valid_idx]
    clean_valid_idx = [idx for idx in valid_idx if idx not in train_set]
    clean_valid_positions = [i for i, idx in enumerate(valid_idx) if idx not in train_set]

    # 2. Load dataset splits
    with open(f'{data_dir}/train_data.pkl', 'rb') as f:
        train_risk, train_attr, train_hete, train_hyp, train_label = pickle.load(f)
    with open(f'{data_dir}/validate_data.pkl', 'rb') as f:
        valid_risk, valid_attr, valid_hete, valid_hyp, valid_label = pickle.load(f)
    with open(f'{data_dir}/test_data.pkl', 'rb') as f:
        test_risk, test_attr, test_hete, test_hyp, test_label = pickle.load(f)

    # 3. Clean labels
    train_label = np.array(train_label)
    clean_valid_label = np.array(valid_label)[clean_valid_positions]
    clean_test_label = np.array(test_label)[clean_test_positions]

    # 4. Load metapath2vec embeddings
    with open(f'{data_dir}/meta_emb.pkl', 'rb') as f:
        com_emb, per_emb = pickle.load(f)

    total_company_num = 3976
    person_num = 2405

    # 5. Build Hierarchical Hypergraphs:
    # Area (Level 1: cities, Level 2: provinces)
    area_keys = list(train_hyp['area'].keys())
    N_cities = len(area_keys)
    city_prov_idx, N_provs = build_city_to_province_map(area_keys)
    
    H_com_area = np.zeros((total_company_num, N_cities), dtype=np.float32)
    for c_idx, k in enumerate(area_keys):
        for comp in train_hyp['area'][k]:
            if comp < total_company_num:
                H_com_area[comp, c_idx] = 1.0

    H_env_area = np.zeros((N_cities, N_provs), dtype=np.float32)
    for c_idx, p_idx in enumerate(city_prov_idx):
        H_env_area[c_idx, p_idx] = 1.0

    # Industry (Level 1: sub-industries, Level 2: sectors)
    ind_keys = list(train_hyp['industry'].keys())
    N_inds = len(ind_keys)
    ind_sec_idx, N_secs = build_industry_to_sector_map(ind_keys)

    H_com_ind = np.zeros((total_company_num, N_inds), dtype=np.float32)
    for i_idx, k in enumerate(ind_keys):
        for comp in train_hyp['industry'][k]:
            if comp < total_company_num:
                H_com_ind[comp, i_idx] = 1.0

    H_env_ind = np.zeros((N_inds, N_secs), dtype=np.float32)
    for i_idx, s_idx in enumerate(ind_sec_idx):
        H_env_ind[i_idx, s_idx] = 1.0

    # 6. Extract Meta-paths and compute data-driven weights (Section 4.2)
    edges = np.array(train_hete[0])
    edge_types = np.array(train_hete[1])

    adj = defaultdict(lambda: defaultdict(list))
    for (u, v), t in zip(edges, edge_types):
        if t < 12:
            adj[t][u].append(v)

    meta_path_defs = [
        ('P_invest_CC', [(6,)]),
        ('P_invested_CC', [(7,)]),
        ('P_branch', [(10,)]),
        ('P_branched', [(11,)]),
        ('P_sup', [(0, 1)]),
        ('P_exec', [(2, 3)]),
        ('P_stk', [(4, 5)]),
        ('P_invP', [(9, 8)]),
        ('P_exec_sup', [(2, 1)]),
        ('P_sup_exec', [(0, 3)]),
        ('P_stk_sup', [(4, 1)]),
        ('P_stk_exec', [(4, 3)]),
    ]

    def get_instances(seq):
        inst = []
        if len(seq) == 1:
            rel = seq[0]
            for u in adj[rel]:
                for v in adj[rel][u]:
                    inst.append((u, v))
        elif len(seq) == 2:
            r1, r2 = seq
            for u in adj[r1]:
                for mid in adj[r1][u]:
                    for v in adj[r2].get(mid, []):
                        if u != v:
                            inst.append((u, v))
        return inst

    meta_path_instances = [get_instances(seqs[0]) for _, seqs in meta_path_defs]
    num_paths = len(meta_path_instances)

    # Global weights s_P^{GL} (Eq. 2)
    node_label = {idx: lbl for idx, lbl in zip(train_idx, train_label)}
    gl_scores = []
    for inst in meta_path_instances:
        rr = nn = rn = nr = 0
        for u, v in inst:
            lu = node_label.get(u, None)
            lv = node_label.get(v, None)
            if lu is not None and lv is not None:
                if lu == 1 and lv == 1: rr += 1
                elif lu == 0 and lv == 0: nn += 1
                elif lu == 1 and lv == 0: rn += 1
                elif lu == 0 and lv == 1: nr += 1
        tot = rr + nn + rn + nr
        val = ((rr + nn) / tot) * (abs(rr - nn) / tot) if tot > 0 else 0.0
        gl_scores.append(val)
    gl_weights = np.exp(gl_scores) / np.sum(np.exp(gl_scores))

    # Type-specific weights s_{i,P}^{TS} with threshold t (Eq. 3, 4, 5)
    all_env_hyperedges = []
    for k in area_keys:
        all_env_hyperedges.append(set(train_hyp['area'][k]))
    for k in ind_keys:
        all_env_hyperedges.append(set(train_hyp['industry'][k]))

    target_map = [defaultdict(list) for _ in meta_path_instances]
    for p_idx, inst in enumerate(meta_path_instances):
        for inst_idx, (u, v) in enumerate(inst):
            target_map[p_idx][v].append(inst_idx)

    env_path_scores = np.zeros((len(all_env_hyperedges), num_paths), dtype=np.float32)
    for e_idx, members in enumerate(all_env_hyperedges):
        for p_idx, inst in enumerate(meta_path_instances):
            rr = nn = rn = nr = 0
            for v in members:
                for inst_idx in target_map[p_idx].get(v, []):
                    u, _ = inst[inst_idx]
                    lu = node_label.get(u, None)
                    lv = node_label.get(v, None)
                    if lu is not None and lv is not None:
                        if lu == 1 and lv == 1: rr += 1
                        elif lu == 0 and lv == 0: nn += 1
                        elif lu == 1 and lv == 0: rn += 1
                        elif lu == 0 and lv == 1: nr += 1
            tot = rr + nn + rn + nr
            if tot > 0:
                val = ((rr + nn) / tot) * (abs(rr - nn) / tot)
                if val >= threshold_t:
                    env_path_scores[e_idx, p_idx] = val

    ent_to_env = defaultdict(list)
    for e_idx, members in enumerate(all_env_hyperedges):
        for m in members:
            ent_to_env[m].append(e_idx)

    ts_weights = np.zeros((total_company_num, num_paths), dtype=np.float32)
    for c_i in range(total_company_num):
        env_list = ent_to_env.get(c_i, [])
        if len(env_list) > 0:
            max_scores = np.max(env_path_scores[env_list, :], axis=0)
        else:
            max_scores = np.zeros(num_paths, dtype=np.float32)
        exp_sc = np.exp(max_scores)
        ts_weights[c_i] = exp_sc / np.sum(exp_sc)

    # Build meta-path adjacency matrices for node-level aggregation
    meta_path_adjs = []
    for inst in meta_path_instances:
        # Create sparse adj matrix
        adj_mat = defaultdict(set)
        for u, v in inst:
            if u < total_company_num and v < total_company_num:
                adj_mat[u].add(v)
        meta_path_adjs.append(adj_mat)

    # 7. Normalize enterprise numerical attributes
    # Build complete company attributes array (3976, 3)
    all_attr = np.zeros((total_company_num, 3), dtype=np.float32)
    for c_idx, attr in zip(train_idx, train_attr):
        all_attr[c_idx] = attr
    for c_idx, attr in zip(valid_idx, valid_attr):
        all_attr[c_idx] = attr
    for c_idx, attr in zip(test_idx, test_attr):
        all_attr[c_idx] = attr

    # Log-transform and standardize attributes
    log_attr = np.log1p(np.maximum(all_attr, 0))
    attr_mean = log_attr[train_idx].mean(axis=0, keepdims=True)
    attr_std = log_attr[train_idx].std(axis=0, keepdims=True) + 1e-6
    norm_attr = (log_attr - attr_mean) / attr_std

    # Combine lawsuit risk info into full dictionary
    full_risk_data = {}
    full_risk_data.update(train_risk)
    full_risk_data.update(valid_risk)
    full_risk_data.update(test_risk)

    # 8. Build full 23-dimensional internal feature matrix
    from utils import initializae_company_info
    full_x = np.zeros((total_company_num, 23), dtype=np.float32)
    x_tr = initializae_company_info(train_risk, train_attr, len(train_idx), 11, 4, 4, train_idx)
    for i, idx in enumerate(train_idx):
        full_x[idx] = x_tr[i]
    x_va = initializae_company_info(valid_risk, valid_attr, len(valid_idx), 11, 4, 4, valid_idx)
    for i, idx in enumerate(valid_idx):
        full_x[idx] = x_va[i]
    x_te = initializae_company_info(test_risk, test_attr, len(test_idx), 11, 4, 4, test_idx)
    for i, idx in enumerate(test_idx):
        full_x[idx] = x_te[i]

    # Standardize full_x based on training set statistics
    x_mean = full_x[train_idx].mean(axis=0, keepdims=True)
    x_std = full_x[train_idx].std(axis=0, keepdims=True) + 1e-6
    norm_x = (full_x - x_mean) / x_std

    return {
        'train_idx': np.array(train_idx),
        'valid_idx': np.array(clean_valid_idx),
        'test_idx': np.array(clean_test_idx),
        'train_label': train_label,
        'valid_label': clean_valid_label,
        'test_label': clean_test_label,
        'clean_test_positions': clean_test_positions,
        'com_emb': com_emb,
        'per_emb': per_emb,
        'norm_attr': norm_attr,
        'norm_x': norm_x.astype(np.float32),
        'full_risk_data': full_risk_data,
        'hete_edges': edges,
        'hete_edge_types': edge_types,
        'H_com_area': H_com_area,
        'H_env_area': H_env_area,
        'H_com_ind': H_com_ind,
        'H_env_ind': H_env_ind,
        'gl_weights': gl_weights.astype(np.float32),
        'ts_weights': ts_weights,
        'meta_path_adjs': meta_path_adjs,
        'total_company_num': total_company_num,
        'person_num': person_num
    }
