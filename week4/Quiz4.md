![image-20200921162813312](C:\Users\123\AppData\Roaming\Typora\typora-user-images\image-20200921162813312.png)

![image-20200921163037739](C:\Users\123\AppData\Roaming\Typora\typora-user-images\image-20200921163037739.png)

![image-20200921163043508](C:\Users\123\AppData\Roaming\Typora\typora-user-images\image-20200921163043508.png)

![image-20200921170156430](C:\Users\123\AppData\Roaming\Typora\typora-user-images\image-20200921170156430.png)

```python
def result():
    s = 'ACAABAACAAABACDBADDDFSDDDFFSSSASDAFAAACBAAAFASD'

    result = []
    # compete the pattern below
    pattern = '([A-Z]*?)(A{3})'
    for item in re.finditer(pattern, s):
      # identify the group number below.
      result.append(item.group(1))
      
    return result
```

![image-20200921170424654](C:\Users\123\AppData\Roaming\Typora\typora-user-images\image-20200921170424654.png)

![image-20200921170430569](C:\Users\123\AppData\Roaming\Typora\typora-user-images\image-20200921170430569.png)

![image-20200921171016249](C:\Users\123\AppData\Roaming\Typora\typora-user-images\image-20200921171016249.png)

![image-20200921171026921](C:\Users\123\AppData\Roaming\Typora\typora-user-images\image-20200921171026921.png)

![image-20200921171104065](C:\Users\123\AppData\Roaming\Typora\typora-user-images\image-20200921171104065.png)

![image-20200921171212662](C:\Users\123\AppData\Roaming\Typora\typora-user-images\image-20200921171212662.png)

![image-20200921171436261](C:\Users\123\AppData\Roaming\Typora\typora-user-images\image-20200921171436261.png)

![image-20200921171615568](C:\Users\123\AppData\Roaming\Typora\typora-user-images\image-20200921171615568.png)

![image-20200921171622890](C:\Users\123\AppData\Roaming\Typora\typora-user-images\image-20200921171622890.png)

![image-20200921172000248](C:\Users\123\AppData\Roaming\Typora\typora-user-images\image-20200921172000248.png)

![image-20200921172009300](C:\Users\123\AppData\Roaming\Typora\typora-user-images\image-20200921172009300.png)