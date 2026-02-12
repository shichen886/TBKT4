document.addEventListener('DOMContentLoaded', function() {
    const enterButton = document.getElementById('enterButton');
    const waterDropOverlay = document.getElementById('waterDropOverlay');
    
    enterButton.addEventListener('click', function(e) {
        const rect = enterButton.getBoundingClientRect();
        const centerX = rect.left + rect.width / 2;
        const centerY = rect.top + rect.height / 2;
        
        createWaterDrop(centerX, centerY);
    });
    
    function createWaterDrop(x, y) {
        waterDropOverlay.style.display = 'block';
        waterDropOverlay.innerHTML = '';
        
        const drop = document.createElement('div');
        drop.className = 'water-drop';
        drop.style.left = x + 'px';
        drop.style.top = y + 'px';
        drop.style.width = '0px';
        drop.style.height = '0px';
        
        waterDropOverlay.appendChild(drop);
        
        gsap.to(drop, {
            width: '300vmax',
            height: '300vmax',
            duration: 1.2,
            ease: 'power2.out',
            onComplete: function() {
                gsap.to(waterDropOverlay, {
                    opacity: 0,
                    duration: 0.5,
                    ease: 'power2.inOut',
                    onComplete: function() {
                        window.location.href = '/dashboard';
                    }
                });
            }
        });
    }
    
    enterButton.addEventListener('mouseenter', function() {
        gsap.to(this, {
            scale: 1.05,
            duration: 0.3,
            ease: 'power2.out'
        });
    });
    
    enterButton.addEventListener('mouseleave', function() {
        gsap.to(this, {
            scale: 1,
            duration: 0.3,
            ease: 'power2.out'
        });
    });
});
