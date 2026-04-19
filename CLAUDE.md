1. Always follow the Google coding style
2. Make the code consummable as readibility is more important than complexity
3. Don't make functions within functions
3. Modularize code and break it down to small functions with LoC keeping the code at the same abstraction level
4. Prioritize readability wherever possible
5. In logging, always log the relevant variables. For example in python, log it as log.info(f"{var=}")
6. If a function is returning more than two elements, then use NamedTuple for return
7. After completion of the task, when you are summarizing what changes you did, please make sure to explain it well and the engineering concepts that was used so that I can learn more
8. Don't write import inside functions just to solve circular imports