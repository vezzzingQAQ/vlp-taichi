## 初始化

```python
import taichi as ti
ti.init(arch=ti.gpu)
```

### ti.init()

![](./../assets/8.png)

### python scope & taichi scope

![](./../assets/9.png)

python-scope中就是常规python代码

### python scope

![](./../assets/10.png)

### taichi scope

![](./../assets/11.png)

被@ti.kernel或者@ti.func修饰的函数

2024.2.2
