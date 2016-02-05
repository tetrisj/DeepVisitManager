import numpy as np
from hashlib import md5


def url_digest(url_str):
    # TODO: This is stupid. Change to something meningful
    h = md5(url_str)
    return float(int(h.hexdigest()[:8], 16))


def read_line(line_str):
    fields = line_str.split('\t')
    timestamp = float(fields[0])
    user_id = fields[3]
    referrer_url = fields[10]
    prev_url = fields[11]
    request_url = fields[12]

    referrer_id = url_digest(referrer_url)
    prev_id = url_digest(prev_url)
    request_id = url_digest(request_url)

    desc = np.array([timestamp, request_id, referrer_id, prev_id])
    return user_id, desc

def main():
    s= '1447088383.953	1560	chrome	0I3h95H8iKrsMDX	186665264191105950	106.44.112.145	156	Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/43.0.2357.124 Safari/537.36 QQBrowser/9.0.2315.400	0	0		http://www.baidu.com/s?ie=utf-8&f=8&rsv_bp=1&ch=2&tn=98010089_dg&wd=%E4%B8%80%E7%AB%99%E5%88%B0%E5%BA%952012&oq=%E4%B8%80%E7%AB%99%E5%88%B0%E5%BA%9520120%26lt%3B02%E6%9C%9F&rsv_pq=fcd25dbb00013555&rsv_t=53ceDrjgIY7YdPiQ0QULMX%2FgGVfrC4xXi%2FS9azzDzv%2FR5M2CtGHszSwhvqVHMQeOav8&rsv_enter=1&rsv_sug3=37&rsv_sug1=34&bs=%E4%B8%80%E7%AB%99%E5%88%B0%E5%BA%9520120302%E6%9C%9F	http://www.baidu.com/s?ie=utf-8&f=8&rsv_bp=1&ch=2&tn=98010089_dg&wd=%E4%B8%80%E7%AB%99%E5%88%B0%E5%BA%95%E9%A2%98%E5%BA%93%E5%8F%8A%E7%AD%94%E6%A1%88&oq=%E4%B8%80%E7%AB%99%E5%88%B0%E5%BA%952012&rsv_pq=8968b96e00012abb&rsv_t=fad7eCTKdKtaOIwlnhiYEmijYs%2B7Jg4TeDAegLYQ%2Bk%2FNcMkzARd%2FyG6k8TwJpRtPhh8&rsv_enter=1&rsv_sug1=38&bs=%E4%B8%80%E7%AB%99%E5%88%B0%E5%BA%952012	0								4015		7899c873995bec6f0148e3447b564201'
    print read_line(s)

if __name__ == '__main__':
    main()
