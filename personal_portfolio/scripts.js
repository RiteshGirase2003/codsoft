    // intro - name 
    var i=0;
    var colors = ['red', 'blue', 'green', 'orange', 'purple'];
    function changeColor() {
        var h1Element = document.getElementById('changingH1');
        i++;
        var glow = colors[i];
        h1Element.style.textShadow = "1px 1px 4px "+glow; 
        if (i==5)
        {
            i=0;
        }
      
    }

    setInterval(changeColor, 1000);

    //  progress bar 

    function   initializeProgressBars()
    {
        proficiency('webDev', 80);
        proficiency('pythonDev', 70); 
    }  

    function proficiency(id, value) 
    {
      var progressBar = document.getElementById(id);
      var width = 0;
    
      var interval = setInterval(function() {
                                                if (width >= value) 
                                                {
                                                    clearInterval(interval);
                                                } 
                                                else 
                                                {
                                                    width++;
                                                    progressBar.style.width = width + '%';
                                                }
                                            }, 10); 
    }
