#Question 2 

#part 1
def function_value(x):
    return 3*(x**2)

#part 2 
def function_derivative(x):
    return (6*x)

#part 3 
def calculate_delta_value(alpha, gradient):
    return -alpha * gradient

#part 4
def gradient_check(gradient,theta):
    return abs(gradient) < theta #boolean values returned 

#part 5 
def gradient_descent(x_init, alpha, theta, max_iters):

    #x_init = starting point 
    #alpha = learning rate
    #theta = threshold for gradient check
    x = x_init
    iterations = 0 

    while iterations < max_iters: 
        #loop will run until iterations = 1000 or gradient check is satisfied - minimum gradient is reached 

        #calculation of the current values of y and gradient
        curr_val = function_value(x)
        gradient = function_derivative(x)

        print(f"Iteration {iterations}: x = {x:.6f}, f(x) = {curr_val:.6f}, f'(x) = {gradient:.6f}")

        if gradient_check(gradient, theta):
            print(f"Converged after {iterations} iterations!")
            print(f"Minimum found at x = {x:.6f}, f(x) = {curr_val:.6f}")
            return x, curr_val, iterations
        
        #update x value using the calculated delta
        delta = calculate_delta_value(alpha, gradient)
        x += delta

        iterations += 1
    
    #if max iterations reached without convergence
    final_value = function_value(x)
    print(f"Reached maximum iterations ({max_iters})")
    print(f"Final result: x = {x:.6f}, f(x) = {final_value:.6f}")
    return x, final_value, iterations

# Set the parameters
print("\nTest 1: Starting from x = 2")
result_x, result_fx, iters = gradient_descent(x_init=2, alpha=0.1, theta=0.001, max_iters=1000)
    
#Test case 2: Start from x = -3
print("\nTest 2: Starting from x = -3")
result_x2, result_fx2, iters2 = gradient_descent(x_init=-3, alpha=0.1, theta=0.001, max_iters=1000)
    
# Test case 3: Smaller learning rate
print("\nTest 3: Smaller learning rate (alpha=0.01)")
result_x3, result_fx3, iters3 = gradient_descent(x_init=2,alpha=0.01, theta=0.001, max_iters=1000)