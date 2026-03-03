SELECT root, min(valid_from) AS first_date, max(valid_to) AS last_date
FROM md.contracts
WHERE dataset='GLBX.MDP3' AND is_spread=false
GROUP BY root
ORDER BY first_date;