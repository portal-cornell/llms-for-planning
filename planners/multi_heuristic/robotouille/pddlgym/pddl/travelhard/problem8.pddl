
(define (problem travel) (:domain travel)
  (:objects
        car-0 - car
	car-1 - car
	car-2 - car
	car-3 - car
	ia - state
	id - state
	il - state
	mi - state
	mn - state
	mo - state
	nd - state
	plane-0 - plane
	sd - state
	wa - state
	wn - state
  )
  (:goal (and
	(visited wa)
	(not (visited mo))
	(not (visited mn))
	(not (visited wn))))
  (:init 
	(drive ia ia car-0)
	(drive ia ia car-1)
	(drive ia ia car-2)
	(drive ia ia car-3)
	(drive ia id car-0)
	(drive ia id car-1)
	(drive ia id car-2)
	(drive ia id car-3)
	(drive ia il car-0)
	(drive ia il car-1)
	(drive ia il car-2)
	(drive ia il car-3)
	(drive ia mi car-0)
	(drive ia mi car-1)
	(drive ia mi car-2)
	(drive ia mi car-3)
	(drive ia mn car-0)
	(drive ia mn car-1)
	(drive ia mn car-2)
	(drive ia mn car-3)
	(drive ia mo car-0)
	(drive ia mo car-1)
	(drive ia mo car-2)
	(drive ia mo car-3)
	(drive ia nd car-0)
	(drive ia nd car-1)
	(drive ia nd car-2)
	(drive ia nd car-3)
	(drive ia sd car-0)
	(drive ia sd car-1)
	(drive ia sd car-2)
	(drive ia sd car-3)
	(drive ia wa car-0)
	(drive ia wa car-1)
	(drive ia wa car-2)
	(drive ia wa car-3)
	(drive ia wn car-0)
	(drive ia wn car-1)
	(drive ia wn car-2)
	(drive ia wn car-3)
	(drive id ia car-0)
	(drive id ia car-1)
	(drive id ia car-2)
	(drive id ia car-3)
	(drive id id car-0)
	(drive id id car-1)
	(drive id id car-2)
	(drive id id car-3)
	(drive id il car-0)
	(drive id il car-1)
	(drive id il car-2)
	(drive id il car-3)
	(drive id mi car-0)
	(drive id mi car-1)
	(drive id mi car-2)
	(drive id mi car-3)
	(drive id mn car-0)
	(drive id mn car-1)
	(drive id mn car-2)
	(drive id mn car-3)
	(drive id mo car-0)
	(drive id mo car-1)
	(drive id mo car-2)
	(drive id mo car-3)
	(drive id nd car-0)
	(drive id nd car-1)
	(drive id nd car-2)
	(drive id nd car-3)
	(drive id sd car-0)
	(drive id sd car-1)
	(drive id sd car-2)
	(drive id sd car-3)
	(drive id wa car-0)
	(drive id wa car-1)
	(drive id wa car-2)
	(drive id wa car-3)
	(drive id wn car-0)
	(drive id wn car-1)
	(drive id wn car-2)
	(drive id wn car-3)
	(drive il ia car-0)
	(drive il ia car-1)
	(drive il ia car-2)
	(drive il ia car-3)
	(drive il id car-0)
	(drive il id car-1)
	(drive il id car-2)
	(drive il id car-3)
	(drive il il car-0)
	(drive il il car-1)
	(drive il il car-2)
	(drive il il car-3)
	(drive il mi car-0)
	(drive il mi car-1)
	(drive il mi car-2)
	(drive il mi car-3)
	(drive il mn car-0)
	(drive il mn car-1)
	(drive il mn car-2)
	(drive il mn car-3)
	(drive il mo car-0)
	(drive il mo car-1)
	(drive il mo car-2)
	(drive il mo car-3)
	(drive il nd car-0)
	(drive il nd car-1)
	(drive il nd car-2)
	(drive il nd car-3)
	(drive il sd car-0)
	(drive il sd car-1)
	(drive il sd car-2)
	(drive il sd car-3)
	(drive il wa car-0)
	(drive il wa car-1)
	(drive il wa car-2)
	(drive il wa car-3)
	(drive il wn car-0)
	(drive il wn car-1)
	(drive il wn car-2)
	(drive il wn car-3)
	(drive mi ia car-0)
	(drive mi ia car-1)
	(drive mi ia car-2)
	(drive mi ia car-3)
	(drive mi id car-0)
	(drive mi id car-1)
	(drive mi id car-2)
	(drive mi id car-3)
	(drive mi il car-0)
	(drive mi il car-1)
	(drive mi il car-2)
	(drive mi il car-3)
	(drive mi mi car-0)
	(drive mi mi car-1)
	(drive mi mi car-2)
	(drive mi mi car-3)
	(drive mi mn car-0)
	(drive mi mn car-1)
	(drive mi mn car-2)
	(drive mi mn car-3)
	(drive mi mo car-0)
	(drive mi mo car-1)
	(drive mi mo car-2)
	(drive mi mo car-3)
	(drive mi nd car-0)
	(drive mi nd car-1)
	(drive mi nd car-2)
	(drive mi nd car-3)
	(drive mi sd car-0)
	(drive mi sd car-1)
	(drive mi sd car-2)
	(drive mi sd car-3)
	(drive mi wa car-0)
	(drive mi wa car-1)
	(drive mi wa car-2)
	(drive mi wa car-3)
	(drive mi wn car-0)
	(drive mi wn car-1)
	(drive mi wn car-2)
	(drive mi wn car-3)
	(drive mn ia car-0)
	(drive mn ia car-1)
	(drive mn ia car-2)
	(drive mn ia car-3)
	(drive mn id car-0)
	(drive mn id car-1)
	(drive mn id car-2)
	(drive mn id car-3)
	(drive mn il car-0)
	(drive mn il car-1)
	(drive mn il car-2)
	(drive mn il car-3)
	(drive mn mi car-0)
	(drive mn mi car-1)
	(drive mn mi car-2)
	(drive mn mi car-3)
	(drive mn mn car-0)
	(drive mn mn car-1)
	(drive mn mn car-2)
	(drive mn mn car-3)
	(drive mn mo car-0)
	(drive mn mo car-1)
	(drive mn mo car-2)
	(drive mn mo car-3)
	(drive mn nd car-0)
	(drive mn nd car-1)
	(drive mn nd car-2)
	(drive mn nd car-3)
	(drive mn sd car-0)
	(drive mn sd car-1)
	(drive mn sd car-2)
	(drive mn sd car-3)
	(drive mn wa car-0)
	(drive mn wa car-1)
	(drive mn wa car-2)
	(drive mn wa car-3)
	(drive mn wn car-0)
	(drive mn wn car-1)
	(drive mn wn car-2)
	(drive mn wn car-3)
	(drive mo ia car-0)
	(drive mo ia car-1)
	(drive mo ia car-2)
	(drive mo ia car-3)
	(drive mo id car-0)
	(drive mo id car-1)
	(drive mo id car-2)
	(drive mo id car-3)
	(drive mo il car-0)
	(drive mo il car-1)
	(drive mo il car-2)
	(drive mo il car-3)
	(drive mo mi car-0)
	(drive mo mi car-1)
	(drive mo mi car-2)
	(drive mo mi car-3)
	(drive mo mn car-0)
	(drive mo mn car-1)
	(drive mo mn car-2)
	(drive mo mn car-3)
	(drive mo mo car-0)
	(drive mo mo car-1)
	(drive mo mo car-2)
	(drive mo mo car-3)
	(drive mo nd car-0)
	(drive mo nd car-1)
	(drive mo nd car-2)
	(drive mo nd car-3)
	(drive mo sd car-0)
	(drive mo sd car-1)
	(drive mo sd car-2)
	(drive mo sd car-3)
	(drive mo wa car-0)
	(drive mo wa car-1)
	(drive mo wa car-2)
	(drive mo wa car-3)
	(drive mo wn car-0)
	(drive mo wn car-1)
	(drive mo wn car-2)
	(drive mo wn car-3)
	(drive nd ia car-0)
	(drive nd ia car-1)
	(drive nd ia car-2)
	(drive nd ia car-3)
	(drive nd id car-0)
	(drive nd id car-1)
	(drive nd id car-2)
	(drive nd id car-3)
	(drive nd il car-0)
	(drive nd il car-1)
	(drive nd il car-2)
	(drive nd il car-3)
	(drive nd mi car-0)
	(drive nd mi car-1)
	(drive nd mi car-2)
	(drive nd mi car-3)
	(drive nd mn car-0)
	(drive nd mn car-1)
	(drive nd mn car-2)
	(drive nd mn car-3)
	(drive nd mo car-0)
	(drive nd mo car-1)
	(drive nd mo car-2)
	(drive nd mo car-3)
	(drive nd nd car-0)
	(drive nd nd car-1)
	(drive nd nd car-2)
	(drive nd nd car-3)
	(drive nd sd car-0)
	(drive nd sd car-1)
	(drive nd sd car-2)
	(drive nd sd car-3)
	(drive nd wa car-0)
	(drive nd wa car-1)
	(drive nd wa car-2)
	(drive nd wa car-3)
	(drive nd wn car-0)
	(drive nd wn car-1)
	(drive nd wn car-2)
	(drive nd wn car-3)
	(drive sd ia car-0)
	(drive sd ia car-1)
	(drive sd ia car-2)
	(drive sd ia car-3)
	(drive sd id car-0)
	(drive sd id car-1)
	(drive sd id car-2)
	(drive sd id car-3)
	(drive sd il car-0)
	(drive sd il car-1)
	(drive sd il car-2)
	(drive sd il car-3)
	(drive sd mi car-0)
	(drive sd mi car-1)
	(drive sd mi car-2)
	(drive sd mi car-3)
	(drive sd mn car-0)
	(drive sd mn car-1)
	(drive sd mn car-2)
	(drive sd mn car-3)
	(drive sd mo car-0)
	(drive sd mo car-1)
	(drive sd mo car-2)
	(drive sd mo car-3)
	(drive sd nd car-0)
	(drive sd nd car-1)
	(drive sd nd car-2)
	(drive sd nd car-3)
	(drive sd sd car-0)
	(drive sd sd car-1)
	(drive sd sd car-2)
	(drive sd sd car-3)
	(drive sd wa car-0)
	(drive sd wa car-1)
	(drive sd wa car-2)
	(drive sd wa car-3)
	(drive sd wn car-0)
	(drive sd wn car-1)
	(drive sd wn car-2)
	(drive sd wn car-3)
	(drive wa ia car-0)
	(drive wa ia car-1)
	(drive wa ia car-2)
	(drive wa ia car-3)
	(drive wa id car-0)
	(drive wa id car-1)
	(drive wa id car-2)
	(drive wa id car-3)
	(drive wa il car-0)
	(drive wa il car-1)
	(drive wa il car-2)
	(drive wa il car-3)
	(drive wa mi car-0)
	(drive wa mi car-1)
	(drive wa mi car-2)
	(drive wa mi car-3)
	(drive wa mn car-0)
	(drive wa mn car-1)
	(drive wa mn car-2)
	(drive wa mn car-3)
	(drive wa mo car-0)
	(drive wa mo car-1)
	(drive wa mo car-2)
	(drive wa mo car-3)
	(drive wa nd car-0)
	(drive wa nd car-1)
	(drive wa nd car-2)
	(drive wa nd car-3)
	(drive wa sd car-0)
	(drive wa sd car-1)
	(drive wa sd car-2)
	(drive wa sd car-3)
	(drive wa wa car-0)
	(drive wa wa car-1)
	(drive wa wa car-2)
	(drive wa wa car-3)
	(drive wa wn car-0)
	(drive wa wn car-1)
	(drive wa wn car-2)
	(drive wa wn car-3)
	(drive wn ia car-0)
	(drive wn ia car-1)
	(drive wn ia car-2)
	(drive wn ia car-3)
	(drive wn id car-0)
	(drive wn id car-1)
	(drive wn id car-2)
	(drive wn id car-3)
	(drive wn il car-0)
	(drive wn il car-1)
	(drive wn il car-2)
	(drive wn il car-3)
	(drive wn mi car-0)
	(drive wn mi car-1)
	(drive wn mi car-2)
	(drive wn mi car-3)
	(drive wn mn car-0)
	(drive wn mn car-1)
	(drive wn mn car-2)
	(drive wn mn car-3)
	(drive wn mo car-0)
	(drive wn mo car-1)
	(drive wn mo car-2)
	(drive wn mo car-3)
	(drive wn nd car-0)
	(drive wn nd car-1)
	(drive wn nd car-2)
	(drive wn nd car-3)
	(drive wn sd car-0)
	(drive wn sd car-1)
	(drive wn sd car-2)
	(drive wn sd car-3)
	(drive wn wa car-0)
	(drive wn wa car-1)
	(drive wn wa car-2)
	(drive wn wa car-3)
	(drive wn wn car-0)
	(drive wn wn car-1)
	(drive wn wn car-2)
	(drive wn wn car-3)
	(fly ia plane-0)
	(fly id plane-0)
	(fly il plane-0)
	(fly mi plane-0)
	(fly mn plane-0)
	(fly mo plane-0)
	(fly nd plane-0)
	(fly sd plane-0)
	(fly wa plane-0)
	(fly wn plane-0)
	(walk ia)
	(walk id)
	(walk il)
	(walk mi)
	(walk mn)
	(walk mo)
	(walk nd)
	(walk sd)
	(walk wa)
	(walk wn)
	(adjacent ia il)
	(adjacent ia mn)
	(adjacent ia sd)
	(adjacent ia wn)
	(adjacent id mo)
	(adjacent id wa)
	(adjacent il ia)
	(adjacent mi wn)
	(adjacent mn ia)
	(adjacent mn nd)
	(adjacent mn sd)
	(adjacent mn wn)
	(adjacent mo id)
	(adjacent mo nd)
	(adjacent mo sd)
	(adjacent nd mn)
	(adjacent nd mo)
	(adjacent nd sd)
	(adjacent sd ia)
	(adjacent sd mn)
	(adjacent sd mo)
	(adjacent sd nd)
	(adjacent wa id)
	(adjacent wn ia)
	(adjacent wn mi)
	(adjacent wn mn)
	(at mi)
	(caravailable car-0)
	(caravailable car-1)
	(caravailable car-2)
	(caravailable car-3)
))
        