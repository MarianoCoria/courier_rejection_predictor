select 
    c.order_id
    ,round(1.00*sum(case when c2.taken=0 then 1 else 0 end)/count(distinct c2.order_id),4) as "store_not_taken_rate_2d"
        ,round(1.00*
        sum(case when 
            (JulianDay(c.created_at) - JulianDay(c2.created_at))<1
            and c2.taken=0 
            then 1 else 0 end)
        /sum(case when 
            (JulianDay(c.created_at) - JulianDay(c2.created_at))<1
            then 1 else 0 end)
    ,4) as "store_not_taken_rate_1d"
    ,round(1.00*
        sum(case when 
            (JulianDay(c.created_at) - JulianDay(c2.created_at))<0.5 
            and c2.taken=0 
            then 1 else 0 end)
        /sum(case when 
            (JulianDay(c.created_at) - JulianDay(c2.created_at))<0.5 
            then 1 else 0 end)
    ,4) as "store_not_taken_rate_12h"
    ,round(1.00*
        sum(case when 
            (JulianDay(c.created_at) - JulianDay(c2.created_at))<0.125 
            and c2.taken=0 
            then 1 else 0 end)
        /sum(case when 
            (JulianDay(c.created_at) - JulianDay(c2.created_at))<0.125
            then 1 else 0 end)
    ,4) as "store_not_taken_rate_3h"
    ,round(1.00*
        sum(case when 
            (JulianDay(c.created_at) - JulianDay(c2.created_at))<0.04 
            and c2.taken=0 
            then 1 else 0 end)
        /sum(case when 
            (JulianDay(c.created_at) - JulianDay(c2.created_at))<0.04
            then 1 else 0 end)
    ,4) as "store_not_taken_rate_1h"

from 
    couriers_rejection c
    left outer join couriers_rejection c2 on c2.store_id=c.store_id 
        and c2.order_id<>c.order_id 
        and c2.created_at < c.created_at 
        and (JulianDay(c.created_at) - JulianDay(c2.created_at))<2
where 1=1
group by 
     c.order_id
;