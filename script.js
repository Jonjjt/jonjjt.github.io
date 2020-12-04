$(document).ready(function() {
    clickevent();
})
function clickevent(){
    $('.collapsible').on('click', function(){
        setTimeout($(this).find('.slideshow').scrollLeft(2000), 0);
        setTimeout($(this).find('.slideshow').animate({scrollLeft: 0}, 1000), 1);
    })
}