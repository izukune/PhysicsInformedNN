# PhysicsInformedNN
### 物理学に基づいたニューラルネットワーク
物理学に基づいたニューラルネットワーク(以下PINN)は深層パーセプトロンを使った関数近似の一種である。与えるデータとして初期条件と境界条件と非線形偏微分方程式を用いて人工知能を学習させている。
### KdV
ここでは(1+1)次元の非線形偏微分方程式を境界条件をディリクレ境界条件として解いた。サンプルとしてKdV方程式を解いている。境界条件の学習データを使うことでディリクレ境界を学習させることができるが、もし境界条件を学習させない場合はFreeの境界となる。
### periodic
ここでは(1+1)次元の非線形偏微分方程式を境界条件を周期境界条件として解いた。境界におけるfの値と傾きを使って周期境界条件を表現している。
### Deffusion-eq
(2+1)次元の偏微分方程式を解く。境界条件は設定していない。
