from gym.envs.registration import register

register(
    id="football2d/SelfTraining-v0",
    entry_point="football2d.envs:SelfTraining_v0",
)
register(
    id="football2d/SelfTraining-v1",
    entry_point="football2d.envs:SelfTraining_v1",
)
register(
    id="football2d/SelfTraining-v2",
    entry_point="football2d.envs:SelfTraining_v2",
)
register(
    id="football2d/OneOnOneMatch",
    entry_point="football2d.envs:OneOnOneMatch",
)
