# 创建时间 : 2023/10/6/0015 17:02
# 作者 : 金有朋
# 邮箱 : 3021927598@qq.com
# 文件名 : lsb_mathing_revisited.py
import math
from PIL import Image


def plus(str):
    # zfill()方法将返回一个指定长度的字符串，向右对齐，不足的位置填充0
    return str.zfill(8)


def decimal_to_bin(n):  # 本函数的意思是将一个十进制数转换成二进制的字符串，采用递归的方式
    if n == 0:
        return '0'
    elif n == 1:
        return '1'
    else:
        return decimal_to_bin(n // 2) + str(n % 2)


def lsb(s):  # 该函数用来求得一个字符串的最后一位
    return s[len(s) - 1]


def f(x1, x2):  # 本函数用来实现算法中做提出的的f()函数
    m1 = math.floor(x1 / 2) + x2
    return lsb(decimal_to_bin(m1))


def get_hide(path):  # 本函数用来读取需要隐藏的文件，并将文件内容转换为二进制数
    #   获取需要隐藏的文件内容
    f = open(path, "rb")
    string = ""
    s = f.read()
    print(f's={s}')  # 这个s是一个二进制字节串
    for i in range(len(s)):  # 依照索引一次遍历这个字节串的每个字节然后转换成二进制，并连接起来
        string = string + plus(
            bin(s[i]).replace('ob', ''))  # 由于bin()函数转换二进制后，二进制字符串的前面会有"0b"来表示这个字符串是二进制形式，所以用replace()替换为空
    f.close()
    return string


def func(img, hide, out):
    im = Image.open(img).convert('L')
    width = im.size[0]
    height = im.size[1]
    print(f'原图的宽为：{width}，高为：{height}')
    count = 0  # 通过长度来判断是否完全将隐藏信息的二进制替换到原图中去
    key = get_hide(hide)  # 获取隐藏的信息
    print(f"隐藏信息二进制的长度为: {len(key)}")
    for h in range(0, height):  # 从左上角开始，依次向右找到两个相邻位像素点的像素值
        for w in range(0, width):
            pixel1 = im.getpixel((w, h))
            pixel2 = im.getpixel((w + 1, h))
            if count == len(key):  # 若计数器count的值为所隐藏信息的长度，则说明已经全部隐藏完了，退出循环
                break
            if key[count] == lsb(decimal_to_bin(pixel1)):  # 按照文章中所提出的算法进行隐藏
                if key[count + 1] != f(pixel1, pixel2):
                    im.putpixel((w + 1, h), pixel2 + 1)
                else:
                    im.putpixel((w + 1, h), pixel2)
                im.putpixel((w, h), pixel1)
            else:
                if key[count + 1] == f(pixel1 - 1, pixel2):
                    im.putpixel((w, h), pixel1 - 1)
                else:
                    im.putpixel((w, h), pixel2 + 1)
                im.putpixel((w + 1, h), pixel2)
            count = count + 2
            print(f"count={count}")
    im.show()  # 显示隐藏信息后的图片
    im.save(out)  # 保存到指定路径


#  原图
old = "D:\programing\python-works\复现\lSb_matching_revisited\ocean.png"
#  需要隐藏的信息
hide = "D:\programing\python-works\复现\lSb_matching_revisited\隐藏信息.txt"
#  隐藏后输出的图片路径
out = "D:\programing\python-works\复现\lSb_matching_revisited\include_secret.png"

func(old, hide, out)
