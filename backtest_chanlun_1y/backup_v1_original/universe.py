import json
import urllib.request
from pathlib import Path

CSI300_UNIVERSE = [
    ("sh510300", "沪深300ETF", "etf"),
    ("sh601318", "中国平安", "stock"), ("sh600036", "招商银行", "stock"), ("sh600030", "中信证券", "stock"),
    ("sh601398", "工商银行", "stock"), ("sh601166", "兴业银行", "stock"), ("sz300059", "东方财富", "stock"),
    ("sh601688", "华泰证券", "stock"), ("sh601211", "国泰君安", "stock"), ("sh601601", "中国太保", "stock"),
    ("sh601628", "中国人寿", "stock"), ("sh601288", "农业银行", "stock"), ("sh601988", "中国银行", "stock"),
    ("sh601939", "建设银行", "stock"), ("sh600000", "浦发银行", "stock"), ("sz000001", "平安银行", "stock"),
    ("sz002142", "宁波银行", "stock"), ("sh601995", "中金公司", "stock"), ("sh601066", "中信建投", "stock"),
    ("sh600999", "招商证券", "stock"), ("sz000776", "广发证券", "stock"), ("sh600519", "贵州茅台", "stock"),
    ("sz000858", "五粮液", "stock"), ("sz000333", "美的集团", "stock"), ("sh600887", "伊利股份", "stock"),
    ("sz000651", "格力电器", "stock"), ("sz002352", "顺丰控股", "stock"), ("sz000568", "泸州老窖", "stock"),
    ("sh600809", "山西汾酒", "stock"), ("sz002304", "洋河股份", "stock"), ("sh600690", "海尔智家", "stock"),
    ("sh601888", "中国中免", "stock"), ("sh603288", "海天味业", "stock"), ("sz002714", "牧原股份", "stock"),
    ("sz300498", "温氏股份", "stock"), ("sh600600", "青岛啤酒", "stock"), ("sz300750", "宁德时代", "stock"),
    ("sz002594", "比亚迪", "stock"), ("sh601012", "隆基绿能", "stock"), ("sz300274", "阳光电源", "stock"),
    ("sz300124", "汇川技术", "stock"), ("sh600900", "长江电力", "stock"), ("sh600438", "通威股份", "stock"),
    ("sz002460", "赣锋锂业", "stock"), ("sz002466", "天齐锂业", "stock"), ("sh601633", "长城汽车", "stock"),
    ("sh600104", "上汽集团", "stock"), ("sz000625", "长安汽车", "stock"), ("sh601689", "拓普集团", "stock"),
    ("sz002920", "德赛西威", "stock"), ("sh600406", "国电南瑞", "stock"), ("sh601985", "中国核电", "stock"),
    ("sh600905", "三峡能源", "stock"), ("sh600011", "华能国际", "stock"), ("sh688981", "中芯国际", "stock"),
    ("sh688012", "中微公司", "stock"), ("sh603501", "韦尔股份", "stock"), ("sz002475", "立讯精密", "stock"),
    ("sz000725", "京东方A", "stock"), ("sz002415", "海康威视", "stock"), ("sh688041", "海光信息", "stock"),
    ("sh688256", "寒武纪", "stock"), ("sz002371", "北方华创", "stock"), ("sz300308", "中际旭创", "stock"),
    ("sz300502", "新易盛", "stock"), ("sh601138", "工业富联", "stock"), ("sz000063", "中兴通讯", "stock"),
    ("sz002230", "科大讯飞", "stock"), ("sz000977", "浪潮信息", "stock"), ("sh603986", "兆易创新", "stock"),
    ("sh688008", "澜起科技", "stock"), ("sh688111", "金山办公", "stock"), ("sh600584", "长电科技", "stock"),
    ("sz002049", "紫光国微", "stock"), ("sh600570", "恒生电子", "stock"), ("sh600845", "宝信软件", "stock"),
    ("sz000938", "紫光股份", "stock"), ("sh601728", "中国电信", "stock"), ("sh600941", "中国移动", "stock"),
    ("sh600050", "中国联通", "stock"), ("sh600276", "恒瑞医药", "stock"), ("sz300760", "迈瑞医疗", "stock"),
    ("sh603259", "药明康德", "stock"), ("sz300015", "爱尔眼科", "stock"), ("sh600436", "片仔癀", "stock"),
    ("sh688271", "联影医疗", "stock"), ("sz000538", "云南白药", "stock"), ("sh600196", "复星医药", "stock"),
    ("sz300122", "智飞生物", "stock"), ("sh601899", "紫金矿业", "stock"), ("sh600309", "万华化学", "stock"),
    ("sh601225", "陕西煤业", "stock"), ("sh601088", "中国神华", "stock"), ("sh600028", "中国石化", "stock"),
    ("sh601857", "中国石油", "stock"), ("sh600111", "北方稀土", "stock"), ("sh603799", "华友钴业", "stock"),
    ("sh601600", "中国铝业", "stock"), ("sh600547", "山东黄金", "stock"), ("sh600019", "宝钢股份", "stock"),
    ("sh600585", "海螺水泥", "stock"), ("sh600031", "三一重工", "stock"), ("sz000425", "徐工机械", "stock"),
    ("sh601766", "中国中车", "stock"), ("sh601919", "中远海控", "stock"), ("sh601816", "京沪高铁", "stock"),
    ("sh600893", "航发动力", "stock"), ("sh600760", "中航沈飞", "stock"), ("sz000768", "中航西飞", "stock"),
    ("sh600150", "中国船舶", "stock"), ("sh600048", "保利发展", "stock"), ("sz001979", "招商蛇口", "stock"),
]

ALL_A_EXTRA = [
    ("sh588000", "科创50ETF", "etf"), ("sz159915", "创业板ETF", "etf"),
    ("sh688825", "长鑫科技", "stock"), ("sh688028", "沃尔德", "stock"), ("sz300394", "天孚通信", "stock"),
    ("sz300442", "润泽科技", "stock"), ("sz300033", "同花顺", "stock"), ("sz300014", "亿纬锂能", "stock"),
    ("sz300433", "蓝思科技", "stock"), ("sz300661", "圣邦股份", "stock"), ("sz300782", "卓胜微", "stock"),
    ("sh688521", "芯原股份", "stock"), ("sh688692", "达梦数据", "stock"), ("sh688036", "传音控股", "stock"),
    ("sh688223", "晶科能源", "stock"), ("sh688126", "沪硅产业", "stock"), ("sh688188", "柏楚电子", "stock"),
    ("sh600418", "江淮汽车", "stock"),
    # Historical portfolio_quant_all_a symbols
    ("sz002881", "美格智能", "stock"), ("sh600938", "中国海油", "stock"), ("sh603028", "赛福天", "stock"),
]

def fetch_dynamic_all_a_leaders():
    dynamic = []
    queries = [("amount", 1), ("amount", 2), ("changepercent", 1)]
    for sort_key, page in queries:
        url = f"https://vip.stock.finance.sina.com.cn/quotes_service/api/json_v2.php/Market_Center.getHQNodeData?page={page}&num=80&sort={sort_key}&asc=0&node=hs_a&symbol=&_s_r_a=init"
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            arr = json.loads(urllib.request.urlopen(req, timeout=8).read().decode())
            for item in arr:
                sym = str(item.get("symbol", "")).lower()
                name = str(item.get("name", ""))
                if (sym.startswith("sh") or sym.startswith("sz")) and "ST" not in name and "退" not in name:
                    dynamic.append((sym, name, "stock"))
        except Exception:
            pass
    return dynamic

def get_universes():
    hs300 = list(CSI300_UNIVERSE)
    seen = {s[0] for s in hs300}
    all_a = list(hs300)
    for item in ALL_A_EXTRA + fetch_dynamic_all_a_leaders():
        if item[0] not in seen:
            seen.add(item[0])
            all_a.append(item)
    return hs300, all_a
