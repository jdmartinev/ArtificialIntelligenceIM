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
