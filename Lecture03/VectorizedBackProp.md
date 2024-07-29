## 2 Retropropagación con Tensores

En el contexto de las redes neuronales, una capa $\(f\)$ es típicamente una función de entradas (tensores) $\(x\)$ y pesos $\(w\)$; la salida (tensor) de la capa es entonces $\(y = f(x, w)\)$. La capa $\(f\)$ está típicamente incrustada en una red neuronal grande con una pérdida escalar $\(L\)$.

Durante la retropropagación, asumimos que se nos da $\(\frac{\partial L}{\partial y}\)$ y nuestro objetivo es calcular $\(\frac{\partial L}{\partial x}\)$ y $\(\frac{\partial L}{\partial w}\)$. Por la regla de la cadena sabemos que

$$
\frac{\partial L}{\partial x} = \frac{\partial L}{\partial y} \frac{\partial y}{\partial x}
$$

$$
\frac{\partial L}{\partial w} = \frac{\partial L}{\partial y} \frac{\partial y}{\partial w}
$$

Por lo tanto, una forma de proceder sería formar las Jacobianas (generalizadas) $\(\frac{\partial y}{\partial x}\)$ y $\(\frac{\partial y}{\partial w}\)$ y usar la multiplicación de matrices (generalizadas) para calcular $\(\frac{\partial L}{\partial x}\)$ y $\(\frac{\partial L}{\partial w}\)$.

Sin embargo, hay un problema con este enfoque: las matrices Jacobianas $\(\frac{\partial y}{\partial x}\)$ y $\(\frac{\partial y}{\partial w}\)$ son típicamente demasiado grandes para caber en la memoria.

Como ejemplo concreto, supongamos que $\(f\)$ es una capa lineal que toma como entrada un minibatch de $\(N\)$ vectores, cada uno de dimensión $\(D\)$, y produce un minibatch de $\(N\)$ vectores, cada uno de dimensión $\(M\)$. Entonces $\(x\)$ es una matriz de forma $\(N \times D\)$, $\(w\)$ es una matriz de forma $\(D \times M\), y \(y = f(x, w) = xw\)$ es una matriz de forma $\(N \times M\)$.

La Jacobiana $\(\frac{\partial y}{\partial x}\)$ entonces tiene forma $\((N \times M) \times (N \times D)\)$. En una red neuronal típica podríamos tener $\(N = 64\)$ y $\(M = D = 4096\)$; entonces $\(\frac{\partial y}{\partial x}\)$ consiste en 64 $\(\cdot\) 4096 \(\cdot\) 64 \(\cdot\) 4096\)$ valores escalares; esto es más de 68 mil millones de números; usando punto flotante de 32 bits, esta matriz Jacobiana ocupará 256 GB de memoria para almacenarse. Por lo tanto, es completamente inútil intentar almacenar y manipular explícitamente la matriz Jacobiana.

Sin embargo, resulta que para la mayoría de las capas de redes neuronales comunes, podemos derivar expresiones que calculan el producto $\(\frac{\partial y}{\partial x} \frac{\partial L}{\partial y}\)$ **sin formar explícitamente la matriz Jacobiana** $\(\frac{\partial y}{\partial x}\)$. Aún mejor, podemos derivar típicamente esta expresión sin siquiera calcular una expresión explícita para la Jacobiana $(\frac{\partial y}{\partial x}\)$; en muchos casos podemos trabajar un caso pequeño en papel y luego inferir la fórmula general.

Veamos cómo funciona esto para el caso de la capa lineal $\(f(x, w) = xw\)$. Supongamos $\(N = 1\)$, $\(D = 2\)$, $\(M = 3\)$. Entonces podemos escribir explícitamente

$$
y = \begin{pmatrix} y_{1,1} & y_{1,2} & y_{1,3} \end{pmatrix} = xw
$$

$$
= \begin{pmatrix} x_{1,1} & x_{1,2} \end{pmatrix} \begin{pmatrix} w_{1,1} & w_{1,2} & w_{1,3} \\ w_{2,1} & w_{2,2} & w_{2,3} \end{pmatrix}
$$

$$
= \begin{pmatrix} x_{1,1}w_{1,1} + x_{1,2}w_{2,1} & x_{1,1}w_{1,2} + x_{1,2}w_{2,2} & x_{1,1}w_{1,3} + x_{1,2}w_{2,3} \end{pmatrix}
$$

Durante la retropropagación asumimos que tenemos acceso a $\(\frac{\partial L}{\partial y}\)$, que técnicamente tiene forma $\( (1) \times (N \times M) \)$; sin embargo, por conveniencia notacional pensaremos en él como una matriz de forma $\(N \times M\)$. Entonces podemos escribir

$$
\frac{\partial L}{\partial y} = \begin{pmatrix} dy_{1,1} & dy_{1,2} & dy_{1,3} \end{pmatrix}
$$

Nuestro objetivo ahora es derivar una expresión para $\(\frac{\partial L}{\partial x}\)$ en términos de $\(x\)$, $\(w\)$ y $\(\frac{\partial L}{\partial y}\)$, sin formar explícitamente toda la Jacobiana $\(\frac{\partial y}{\partial x}\)$. Sabemos que $\(\frac{\partial L}{\partial x}\)$ tendrá forma $\( (1) \times (N \times D) \)$, pero como es típico para representar gradientes, en su lugar, vemos $\(\frac{\partial L}{\partial x}\)$ como una matriz de forma $\(N \times D\)$. Sabemos que cada elemento de $\(\frac{\partial L}{\partial x}\)$ es un escalar que da las derivadas parciales de $\(L\)$ con respecto a los elementos de $\(x\)$:

$$
\frac{\partial L}{\partial x} = \begin{pmatrix} \frac{\partial L}{\partial x_{1,1}} & \frac{\partial L}{\partial x_{1,2}} \end{pmatrix}
$$

Pensando en un elemento a la vez, la regla de la cadena nos dice que

$$
\frac{\partial L}{\partial x_{1,1}} = \frac{\partial L}{\partial y} \frac{\partial y}{\partial x_{1,1}}
$$
$$
\frac{\partial L}{\partial x_{1,2}} = \frac{\partial L}{\partial y} \frac{\partial y}{\partial x_{1,2}}
$$

Viendo estas derivadas como matrices generalizadas, $\(\frac{\partial L}{\partial y}\)$ tiene forma $\((1) \times (N \times M)\)$ y $\(\frac{\partial y}{\partial x_{1,1}}\)$ tiene forma $\((N \times M) \times (1)\)$; su producto $\(\frac{\partial L}{\partial x_{1,1}}\)$ entonces tiene forma $\((1) \times (1)\)$. Si en cambio vemos $\(\frac{\partial L}{\partial y}\)$ y $\(\frac{\partial y}{\partial x_{1,1}}\)$ como matrices de forma $\(N \times M\)$, entonces su producto matricial generalizado es simplemente el producto punto $\(\frac{\partial L}{\partial y} \cdot \frac{\partial y}{\partial x_{1,1}}\)$.

Ahora computamos

$$
\frac{\partial y}{\partial x_{1,1}} = \begin{pmatrix} \frac{\partial y_{1,1}}{\partial x_{1,1}} & \frac{\partial y_{1,2}}{\partial x_{1,1}} & \frac{\partial y_{1,3}}{\partial x_{1,1}} \end{pmatrix} = \begin{pmatrix} w_{1,1} & w_{1,2} & w_{1,3} \end{pmatrix}
$$

$$
\frac{\partial y}{\partial x_{1,2}} = \begin{pmatrix} \frac{\partial y_{1,1}}{\partial x_{1,2}} & \frac{\partial y_{1,2}}{\partial x_{1,2}} & \frac{\partial y_{1,3}}{\partial x_{1,2}} \end{pmatrix} = \begin{pmatrix} w_{2,1} & w_{2,2} & w_{2,3} \end{pmatrix}
$$

donde la igualdad final proviene de tomar las derivadas de la Ecuación 3 con respecto a $\(x_{1,1}\)$.

Ahora podemos combinar estos resultados y escribir

$$
\frac{\partial L}{\partial x_{1,1}} = \frac{\partial L}{\partial y} \frac{\partial y}{\partial x_{1,1}} = dy_{1,1} w_{1,1} + dy_{1,2} w_{1,2} + dy_{1,3} w_{1,3}
$$

$$
\frac{\partial L}{\partial x_{1,2}} = \frac{\partial L}{\partial y} \frac{\partial y}{\partial x_{1,2}} = dy_{1,1} w_{2,1} + dy_{1,2} w_{2,2} + dy_{1,3} w_{2,3}
$$

Esto nos da nuestra expresión final para $\(\frac{\partial L}{\partial x}\)$:

$$
\frac{\partial L}{\partial x} = \begin{pmatrix} \frac{\partial L}{\partial x_{1,1}} & \frac{\partial L}{\partial x_{1,2}} \end{pmatrix}
$$
$$
= \begin{pmatrix} dy_{1,1} w_{1,1} + dy_{1,2} w_{1,2} + dy_{1,3} w_{1,3} & dy_{1,1} w_{2,1} + dy_{1,2} w_{2,2} + dy_{1,3} w_{2,3} \end{pmatrix}^T
$$

$$
= \frac{\partial L}{\partial y} x^T
$$

Este resultado final $\(\frac{\partial L}{\partial x} = \frac{\partial L}{\partial y} x^T\)$ es muy interesante porque nos permite calcular $\(\frac{\partial L}{\partial x}\)$ de manera eficiente sin formar explícitamente la Jacobiana $\(\frac{\partial y}{\partial x}\)$. Solo derivamos esta fórmula para el caso específico de $\(N = 1\), \(D = 2\), \(M = 3\)$, pero de hecho tiene generalidad.
