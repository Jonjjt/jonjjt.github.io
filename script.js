let navstate = true;
function togNav() {
    if (navstate) {
        document.getElementById("sidenav").style.width = "200px";
        document.getElementById("dim").style.display = "initial";
        navstate = false;
        console.log(navstate);
    } 
    else {
        document.getElementById("sidenav").style.width = "0";
        document.getElementById("dim").style.display = "none";
        navstate = true;
    }
}
function closeNav() {
    document.getElementById("sidenav").style.width = "0";
    document.getElementById("dim").style.display = "none";
    navstate = true;
}
window.onscroll = function() {
    scrollFunction();
}
function scrollFunction() {
    if (document.body.scrollTop > 55 || document.documentElement.scrollTop > 55) {
        document.getElementById("menubtn2").style.display = "block";
        document.getElementById("sttbtn").style.display = "block";
    } 
    else {
        document.getElementById("menubtn2").style.display = "none";
        document.getElementById("sttbtn").style.display = "none";
    }
}
function scrolltoTop() {
    document.body.scrollTop = 0; // For Safari
    document.documentElement.scrollTop = 0; // For Chrome, Firefox, IE and Opera
  }