

function clearAll() {
    currentInput = '';
    result = 0;
    operator = '';
    document.getElementById('display').value = '';
}

function clearLast() {
    currentInput = currentInput.slice(0, -1);
    document.getElementById('display').value = currentInput;
}


let currentInput = '';
let result = 0;
let operator = '';
let cursorPosition = 0; 
let ans=0;

function appendToDisplay(value) {
    
    if (isOperator(currentInput.charAt(cursorPosition - 1)) && isOperator(value)) {
    
        currentInput = currentInput.slice(0, cursorPosition - 1) + value + currentInput.slice(cursorPosition);
        updateDisplay();
    } else {
    
        currentInput = currentInput.slice(0, cursorPosition) + value + currentInput.slice(cursorPosition);
        cursorPosition += value.length;
        updateDisplay();
    }
}


function isOperator(char) {
    return ['+', '-', '*', '/'].includes(char);
}


function clearDisplay() {
    currentInput = '';
    result = 0;
    operator = '';
    cursorPosition = 0;
    updateDisplay();
}

function calculateResult() {
    try {
        result = eval(currentInput);
        ans=result
        currentInput = result.toString();
        cursorPosition = currentInput.length; 
        updateDisplay();
    } catch (error) {
        currentInput = 'Error';
        cursorPosition = currentInput.length; 
        updateDisplay();
    }
}

function updateDisplay() {
    document.getElementById('display').value = currentInput;
    document.getElementById('display').setSelectionRange(cursorPosition, cursorPosition);
}


function moveCursorBackward() {
    if (cursorPosition > 0) {
        cursorPosition--;
        updateDisplay();
    }
}

function moveCursorForward() {
    if (cursorPosition < currentInput.length) {
        cursorPosition++;
        updateDisplay();
    }
}

function prev_ans()
{
    try {
        currentInput = ans.toString();
        cursorPosition = currentInput.length; 
        updateDisplay();
    } catch (error) {
        currentInput = 'Error';
        cursorPosition = currentInput.length; 
        updateDisplay();
    }

}