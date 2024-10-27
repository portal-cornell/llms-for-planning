(define (problem gripper-5-5-2)
(:domain gripper-strips)
(:objects robot1 robot2 robot3 robot4 robot5 - robot
rgripper1 lgripper1 rgripper2 lgripper2 rgripper3 lgripper3 rgripper4 lgripper4 rgripper5 lgripper5 - gripper
room1 room2 room3 room4 room5 - room
ball1 ball2 - object)
(:init
(at-robby robot1 room3)
(free robot1 rgripper1)
(free robot1 lgripper1)
(at-robby robot2 room2)
(free robot2 rgripper2)
(free robot2 lgripper2)
(at-robby robot3 room2)
(free robot3 rgripper3)
(free robot3 lgripper3)
(at-robby robot4 room3)
(free robot4 rgripper4)
(free robot4 lgripper4)
(at-robby robot5 room2)
(free robot5 rgripper5)
(free robot5 lgripper5)
(at ball1 room1)
(at ball2 room3)
)
(:goal
(and
(at ball1 room5)
(at ball2 room5)
)
)
)
