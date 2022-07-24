import urllib.request
import urllib.parse
import json

def scrape_e621(arg, limit=10):
    print("Doing E621 query with args: %s" % arg)
    query_string=urllib.parse.urlencode({"tags" : arg, "limit": limit})
    html_content=urllib.request.Request("http://e621.net/post/index.json?"+query_string, headers={"User-Agent": 'letstagporn/1.0 (by lionflower on e621)'})
    f=urllib.request.urlopen(html_content)
    print(html_content)
    contents=f.read().decode("utf-8")
    print(contents)
    '''f.close()
    j=json.loads(contents)
    newresult=[]
    for elem in j:
        print("Found elem with score %d" % elem['score'])
        if elem['score']>=0:
            newresult.append(elem)
    return newresult

#def exportImages(results):

'''
if __name__=="__main__":
    images = scrape_e621(["watersports"])
    '''for i in images:
        print(i)
'''