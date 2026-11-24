# SQL queries on `student_cost.db` (Unit III: working with relational databases)

## Q1  Average monthly cost by housing type (JOIN + GROUP BY)
```sql
SELECT s.housing, COUNT(DISTINCT s.id) n, ROUND(SUM(e.amount)*1.0/COUNT(DISTINCT s.id)) avg_cost FROM student s JOIN expense e USING(id) GROUP BY s.housing ORDER BY avg_cost DESC
```

| housing   |   n |   avg_cost |
|:----------|----:|-----------:|
| Flat      |  10 |      25800 |
| PG        |  25 |      22660 |
| Hostel    |  12 |       8750 |
| Home      |   3 |       6500 |

## Q2  Category cost per campus (pivot via CASE)
```sql
SELECT s.campus, ROUND(AVG(CASE WHEN e.category='Rent' THEN e.amount END)) rent, ROUND(AVG(CASE WHEN e.category='Food' THEN e.amount END)) food, ROUND(AVG(CASE WHEN e.category='Commute' THEN e.amount END)) commute, ROUND(AVG(CASE WHEN e.category='Academics' THEN e.amount END)) academics FROM student s JOIN expense e USING(id) GROUP BY s.campus
```

| campus   |   rent |   food |   commute |   academics |
|:---------|-------:|-------:|----------:|------------:|
| Bidholi  |   8732 |   7161 |      2357 |        4196 |
| Kandoli  |   4364 |   5045 |      1295 |        3864 |

## Q3  Costliest localities with at least 3 students (HAVING)
```sql
SELECT s.locality, COUNT(DISTINCT s.id) n, ROUND(SUM(e.amount)*1.0/COUNT(DISTINCT s.id)) avg_cost FROM student s JOIN expense e USING(id) GROUP BY s.locality HAVING n>=3 ORDER BY avg_cost DESC
```

| locality        |   n |   avg_cost |
|:----------------|----:|-----------:|
| Pondha          |   5 |      29500 |
| Nanda Ki Chowki |   8 |      23813 |
| Kandoli         |   6 |      22167 |
| Bidholi         |  11 |      21591 |
| Prem Nagar      |   6 |      20583 |
| Kandoli Hostel  |  10 |       8350 |

## Q4  Students spending more than they receive (subquery)
```sql
SELECT s.id, s.campus, s.housing, s.allowance+s.side_income income, t.spend, t.spend-(s.allowance+s.side_income) shortfall FROM student s JOIN (SELECT id, SUM(amount) spend FROM expense GROUP BY id) t USING(id) WHERE t.spend > s.allowance+s.side_income ORDER BY shortfall DESC
```

|   id | campus   | housing   |   income |   spend |   shortfall |
|-----:|:---------|:----------|---------:|--------:|------------:|
|    8 | Bidholi  | PG        |    15000 |   29000 |       14000 |
|   14 | Bidholi  | PG        |     9500 |   21500 |       12000 |
|   15 | Bidholi  | PG        |    18000 |   25000 |        7000 |
|   43 | Bidholi  | PG        |    20000 |   27000 |        7000 |
|   33 | Bidholi  | PG        |    16000 |   22500 |        6500 |
|   10 | Bidholi  | PG        |    15000 |   20000 |        5000 |
|   38 | Bidholi  | PG        |    26000 |   31000 |        5000 |
|    9 | Bidholi  | PG        |    20000 |   24500 |        4500 |
|   26 | Bidholi  | PG        |    19000 |   23000 |        4000 |
|   28 | Kandoli  | PG        |    25000 |   28500 |        3500 |
|    5 | Kandoli  | Flat      |    17000 |   20000 |        3000 |
|   16 | Kandoli  | PG        |    18000 |   21000 |        3000 |
|   46 | Bidholi  | Hostel    |    12000 |   13500 |        1500 |
|   21 | Bidholi  | PG        |    25000 |   26000 |        1000 |
|   36 | Kandoli  | Hostel    |    10000 |   11000 |        1000 |
|   48 | Kandoli  | PG        |    23000 |   24000 |        1000 |
|    2 | Bidholi  | PG        |    23000 |   23500 |         500 |
|   29 | Kandoli  | PG        |    15000 |   15500 |         500 |
|   31 | Bidholi  | Flat      |    19000 |   19500 |         500 |

## Q5  Stress by year of study (aggregate)
```sql
SELECT year, COUNT(*) n, ROUND(AVG(stress),2) avg_stress, MAX(stress) max_stress FROM student GROUP BY year
```

|   year |   n |   avg_stress |   max_stress |
|-------:|----:|-------------:|-------------:|
|      1 |  15 |         5.8  |           10 |
|      2 |  13 |         6.08 |           10 |
|      3 |  15 |         4.47 |            9 |
|      4 |   7 |         4    |            8 |

## Q6  Rank students within housing by spend (window function)
```sql
SELECT * FROM (SELECT s.id, s.housing, SUM(e.amount) spend, RANK() OVER (PARTITION BY s.housing ORDER BY SUM(e.amount) DESC) rk FROM student s JOIN expense e USING(id) GROUP BY s.id) WHERE rk<=2 ORDER BY housing, rk
```

|   id | housing   |   spend |   rk |
|-----:|:----------|--------:|-----:|
|    4 | Flat      |   43000 |    1 |
|   22 | Flat      |   27000 |    2 |
|   27 | Flat      |   27000 |    2 |
|   18 | Home      |    7500 |    1 |
|   23 | Home      |    6000 |    2 |
|   49 | Home      |    6000 |    2 |
|   46 | Hostel    |   13500 |    1 |
|   36 | Hostel    |   11000 |    2 |
|   45 | PG        |   35500 |    1 |
|   38 | PG        |   31000 |    2 |
