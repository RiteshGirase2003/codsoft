function isElementPartiallyInViewport(el, visibilityPercentage) {
    const rect = el.getBoundingClientRect();
    const threshold = visibilityPercentage / 100;
    return (
        rect.bottom >= (threshold * window.innerHeight) &&
        rect.top <= ((1 - threshold) * window.innerHeight)
    );
}

const image1 = document.querySelector('.img1'); 
const image2 = document.querySelector('.img2'); 
const image3 = document.querySelector('.img3');
const image4 = document.querySelector('.img4'); 
const image5 = document.querySelector('.img5');             
const image6 = document.querySelector('.img6'); 

const textElement1 = document.getElementById('big_data');
const textElement2 = document.getElementById('ai');
const textElement3 = document.getElementById('ds');
const textElement4 = document.getElementById('gd');
const textElement5 = document.getElementById('ui');
const textElement6 = document.getElementById('cloud');

const visibilityPercentage = 60; // Adjust this value for the desired visibility percentage

window.addEventListener('scroll', function () {
    if (isElementPartiallyInViewport(image1, visibilityPercentage)) {
        textElement1.style.display = 'block';
    } else {
        textElement1.style.display = 'none';
    }

    if (isElementPartiallyInViewport(image2, visibilityPercentage)) {
        textElement2.style.display = 'block';
    } else {
        textElement2.style.display = 'none';
    }

    if (isElementPartiallyInViewport(image3, visibilityPercentage)) {
        textElement3.style.display = 'block';
    } else {
        textElement3.style.display = 'none';
    }

    if (isElementPartiallyInViewport(image4, visibilityPercentage)) {
        textElement4.style.display = 'block';
    } else {
        textElement4.style.display = 'none';
    }

    if (isElementPartiallyInViewport(image5, visibilityPercentage)) {
        textElement5.style.display = 'block';
    } else {
        textElement5.style.display = 'none';
    }

    if (isElementPartiallyInViewport(image6, visibilityPercentage)) {
        textElement6.style.display = 'block';
    } else {
        textElement6.style.display = 'none';
    }
});