# Movie Recommender System using AIRL(adverserial inverse reinforcement learning)
This is a PyTorch implementation of Adversarial Inverse Reinforcement Learning(AIRL)[1] and GAIL[2] based on DDPG[2] for mine master thesis. The implementation took the code references of [3]-[10].
Also thank you for Prof. Dr. Heinz Köppl[11] and supervisor[12] giving me this chance to implement my ideas as graduation work.

## Setup
You can install Python liblaries using `pip3 install -r requirements.txt`.

## Example
train forward model:
```
python3 train_rl.py
```
train inverse model:
```
python3 train_imitation.py
```
run online server: extract rswebsite.zip, install maven dependencies in pom.xml
then run **main** function in RecServer. Open http://localhost:6016/ in browser.

Run deployed model via TensorFlow Serving(in windows), docker download link: https://www.docker.com/. Open docker and then:
```
docker run -t --rm -p 8501:8501  -v ".\rswebsite\src\main\resources\webroot\modeldata\rl:/models/recmodel"  -e MODEL_NAME=recmodel tensorflow/serving
```
## References

[[1]](https://arxiv.org/abs/1710.11248) Fu, Justin, Katie Luo, and Sergey Levine. "Learning robust rewards with adversarial inverse reinforcement learning." arXiv preprint arXiv:1710.11248 (2017).\
[[2]](https://arxiv.org/abs/1509.02971) Timothy P. Lillicrap et.al: "Continuous control with deep reinforcement learning".\
[[3]](https://mofanpy.com/) machine learning of mofan\
[[4]](https://arxiv.org/abs/1809.02925) Discriminator-Actor-Critic: Addressing Sample Inefficiency and Reward Bias in Adversarial Imitation Learning\
[[5]](https://github.com/ku2482/gail-airl-ppo.pytorch) gail-airl-ppo.pytorch project by Toshiki Watanabe\
[[6]](https://github.com/Ericonaldo/ILSwiss) ILSwiss is an Easy-to-run Imitation Learning (IL, or Learning from Demonstration, LfD) and also Reinforcement Learning (RL) framework (template) in PyTorch.\
[[7]](https://github.com/marload/DeepRL-TensorFlow2) Simple implementations of various popular Deep Reinforcement Learning algorithms using TensorFlow2\
[[8]](https://github.com/zhengjxu/SparrowRecSys) A Deep Learning Recommender System\
[[9]](https://github.com/backgom2357/Recommender_system_via_deep_RL) Deep Reinforcement Learning based Recommender System by Young_Painter_L,kyunghoon-jung and pasus\
[[10]](https://github.com/hillup/recommend) practice on movielens using pytorch\
[[11]](https://www.bcs.tu-darmstadt.de/team_sos/koepplheinz_sos.en.jsp) Prof. Dr. Heinz Köppl\
[[12]](https://www.bcs.tu-darmstadt.de/team_sos/schultheismatthias_sos.en.jsp) Matthias Schultheis\
[[13]](https://arxiv.org/abs/1801.01290) Haarnoja, Tuomas, et al. "Soft actor-critic: Off-policy maximum entropy deep reinforcement learning with a stochastic actor." arXiv preprint arXiv:1801.01290 (2018).\

