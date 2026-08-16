# No-AI gate

Complete this gate without AI-generated code.

Start from this fresh program:

```python
stock = {"dough": 12, "cheese": 8, "tomato": 10}

def consume(stock, ingredient, quantity):
    stock[ingredient] = stock[ingredient] - quantity
    return stock[ingredient]

print(consume(stock, "cheese", 3))
```

Tasks:

1. Predict the printed value without running it.
2. Modify the function so inventory can never become negative.
3. Make an unknown ingredient produce a clear controlled result rather than an unhandled KeyError.
4. Add a second function that reports ingredients below a configurable reorder threshold.
5. Introduce one small bug deliberately.
6. Use a trace or smallest failing example to diagnose and repair it.
7. Explain every change in plain language.
8. State what evidence shows the repaired program is correct.

Passing requires working code plus explanation and debugging evidence.
