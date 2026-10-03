// Function to update the file name
function updateFileName() {
    const fileInput = document.getElementById('fileInput');
    const fileLabel = document.querySelector('.file-upload-info');

    if (fileInput.files.length > 0) {
        fileLabel.value = fileInput.files[0].name;
    } else {
        fileLabel.value = 'No file chosen';
    }
}

// Event listener for the Cancel button
document.getElementById('cancelButton').addEventListener('click', function () {
    console.log('Cancel button clicked.');
    hideSections();
});


// Function to show the Load Dataset section
function showDashboardSection() {
    hideSections();
    document.getElementById('dashboardOption').style.display = 'block';
    hidepredictSection(); // Hide placement section when clicking on other sections
}

// Function to show the Load Dataset section
function showLoadDatasetSection() {
    hideSections();
    document.getElementById('loadDatasetSection').style.display = 'block';
    hidepredictSection(); // Hide placement section when clicking on other sections
}

// Function to show the Convert to Integers section
function showConvertToIntSection() {
    hideSections();
    document.getElementById('convertToIntSection').style.display = 'block';
    hidepredictSection(); // Hide placement section when clicking on other sections
}

// Function to show the Train and Evaluate section
function showTrainAndEvaluateSection() {
    hideSections();
    document.getElementById('trainAndEvaluateSection').style.display = 'block';
    hidepredictSection(); // Hide placement section when clicking on other sections
}

// Function to show the Placement Prediction section
function showpredictSection() {
    hideSections(); // Hide other sections when opening placement section
    document.getElementById('predictSection').style.display = 'block';
}

// Function to hide all sections
function hideSections() {
    const sections = document.querySelectorAll('.section');
    sections.forEach(section => {
        section.style.display = 'none';
    });
}

// Event listeners for each option

document.getElementById('dashboardOption').addEventListener('click', function () {
    showLoadDatasetSection();
});

document.getElementById('loadDatasetOption').addEventListener('click', function () {
    showLoadDatasetSection();
});

document.getElementById('convertToIntOption').addEventListener('click', function () {
    showConvertToIntSection();
});

document.getElementById('trainAndEvaluateOption').addEventListener('click', function () {
    showTrainAndEvaluateSection();
});

// Event listener for the Placement Prediction option
document.getElementById('predictOption').addEventListener('click', function () {
    showpredictSection();
});


// Function to toggle the EDA dropdown menu
function toggleEdaDropdown() {
    const dropdown = document.getElementById('edaDropdown');
    dropdown.classList.toggle('show'); // Toggle the visibility of the dropdown
}

// Event listener for the EDA option to toggle the dropdown
document.getElementById('edaOption').addEventListener('click', function(event) {
    event.preventDefault(); // Prevent the default action of the anchor element
    toggleEdaDropdown(); // Toggle the visibility of the dropdown
});

// Event listener to close the dropdown if the user clicks outside of it or on other options
document.addEventListener('click', function(event) {
    const dropdown = document.getElementById('edaDropdown');
    if (!event.target.matches('.dropdown-toggle') && !dropdown.contains(event.target)) {
        dropdown.classList.remove('show');
    }
});

// Function to populate dropdown menus with values from the CSV file
function populateDropdownsFromCSV(columnName, dropdownId) {
    var dropdown = document.getElementById(dropdownId);
    // Check if the dropdown already contains options
    if (dropdown.options.length === 0) {
        fetch(`static/transformed_data/${columnName}_details.csv`)
            .then(response => response.text())
            .then(csvData => {
                // Split CSV data by lines and iterate over each line
                csvData.split('\n').forEach(row => {
                    // Split each row into values and add them as options to the dropdown
                    var option = document.createElement('option');
                    option.text = row.trim(); // Trim to remove leading/trailing whitespace
                    dropdown.add(option);
                });
            })
            .catch(error => {
                console.error(`Error fetching ${columnName} details:`, error);
            });
    }
}

// Populate each dropdown menu with values from the corresponding CSV file
populateDropdownsFromCSV('branch', 'branch');
populateDropdownsFromCSV('cohabit', 'cohabit');
populateDropdownsFromCSV('scholarship', 'scholarship');
populateDropdownsFromCSV('hsc_branch', 'hsc_branch');
populateDropdownsFromCSV('caste', 'caste');
populateDropdownsFromCSV('weekly_studytime', 'weekly_studytime');
populateDropdownsFromCSV('extra_classes', 'extra_classes');
populateDropdownsFromCSV('attendance', 'attendance');
populateDropdownsFromCSV('hsc_medium', 'hsc_medium');
populateDropdownsFromCSV('certification', 'certification');
