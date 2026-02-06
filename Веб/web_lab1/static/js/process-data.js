let selectedR = null;

class InvalidValueException extends Error {
    constructor(message) {
        super(message);
        this.name = "InvalidValueException";
    }
}

function selectR(value, button) {
    selectedR = parseFloat(value);
    const rButtons = document.getElementsByName('R-button');
    rButtons.forEach(btn => btn.classList.remove('active'));
    button.classList.add('active');
}

function validateValues() {
    const xRadio = document.querySelector('input[name="X-radio"]:checked');
    const xValue = xRadio ? parseFloat(xRadio.value) : null;
    
    const yValue = parseFloat(document.getElementById('Y-input').value);
    
    const ylen = document.getElementById('Y-input').value.split('.')[1];
    if (ylen && ylen.length > 15) {
        alert("Слишком большое количество знаков после запятой");
        throw new InvalidValueException("Слишком большое количество знаков после запятой");
    }

    if (xValue === null) {
        alert("Ошибка: выберите значение X.");
        throw new InvalidValueException('Ошибка: выберите значение X.');
    }

    if (isNaN(yValue) || yValue < -3 || yValue > 5) {
        alert("Ошибка: значение Y должно быть в диапазоне от -3 до 5.");
        throw new InvalidValueException('Ошибка: значение Y должно быть в диапазоне от -3 до 5.');
    }

    if (selectedR === null) {
        alert("Ошибка: выберите значение R.");
        throw new InvalidValueException('Ошибка: выберите значение R.');
    }

    return true;
}

document.addEventListener("DOMContentLoaded", () => {
    function loadPreviousRequests() {
        const previousRequests = JSON.parse(localStorage.getItem('prev-requests')) || [];
        previousRequests.reverse().forEach(request => {
            printRow(request);
        });
    }

    loadPreviousRequests();

    document.getElementById('process-data').addEventListener('click', function (e) {
        e.preventDefault();
        
        validateValues();

        const xVal = parseFloat(document.querySelector('input[name="X-radio"]:checked').value);
        const yVal = parseFloat(document.querySelector('#Y-input').value);
        const rVal = parseFloat(selectedR);

        fetch(`/fcgi-bin/Web1.jar?x=${xVal}&y=${yVal}&r=${rVal}`, {
            method: 'GET',
        }).then(response => {
            console.log(`x=${xVal}, y=${yVal}, r=${rVal}`);
            if (!response.ok) {
                throw new Error(`${response.status}`);
            }
            return response.json();
        }).then((response) => {
            console.log(response);
            if (response.error) {
                alert(response.error);
            } else {
                printRow(response);
                const previousRequests = JSON.parse(localStorage.getItem('prev-requests')) || [];
                previousRequests.push(response);
                localStorage.setItem('prev-requests', JSON.stringify(previousRequests));
            }
        }).catch((error) => {
            alert(`There was an error processing your request: ${error.message}`);
        });
    });

    function printRow(data) {
        const newRow = document.createElement('tr');

        const xCell = document.createElement('td');
        xCell.textContent = data.x;
        newRow.appendChild(xCell);

        const yCell = document.createElement('td');
        yCell.textContent = data.y;
        newRow.appendChild(yCell);

        const rCell = document.createElement('td');
        rCell.textContent = data.r;
        newRow.appendChild(rCell);

        const resultCell = document.createElement('td');
        if(data.result == 'True') resultCell.textContent = "Point is inside the graph"
        else resultCell.textContent = "Point is outside the graph"

        newRow.appendChild(resultCell);

        const timeCell = document.createElement('td');
        const requestTime = new Date(data.current_time);
        timeCell.textContent = requestTime.toLocaleString('en-GB', {
            year: 'numeric',
            month: 'short',
            day: '2-digit',
            hour: '2-digit',
            minute: '2-digit',
            second: '2-digit'
        });
        newRow.appendChild(timeCell);

        const execTimeCell = document.createElement('td');
        execTimeCell.textContent = data.execution_time;
        newRow.appendChild(execTimeCell);

        const tableBody = document.querySelector('.graph-table table tbody');
        tableBody.insertBefore(newRow, tableBody.firstChild);
    }
});
