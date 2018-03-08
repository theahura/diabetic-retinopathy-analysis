$(document).ready(() => {

	var $panzoom = $('.panzoomed').panzoom();
	$panzoom.parent().on('mousewheel.focal', function( e ) {
		e.preventDefault();
		var delta = e.delta || e.originalEvent.wheelDelta;
		var zoomOut = delta ? delta < 0 : e.originalEvent.deltaY > 0;
		console.log(e.clientX + " " + e.clientY)
		$panzoom.panzoom('zoom', zoomOut, {
			increment: 0.1,
			focal: { clientX: e.clientX, clientY: e.clientY}
		});
	});

	$('.side').click(function() {
		var bgi = $(this).css('background-image');
		bgi = bgi.replace('url(','').replace(')','').replace(/\"/gi, "");
		console.log(bgi)
		$('#selected-image').attr('src', bgi);
		$panzoom.panzoom("reset", {animate: false});
	});

	$("#file-upload").change(() => {
		var input = $('#file-upload')[0];
		if (input.files && input.files[0]) {
			var reader = new FileReader();

			reader.onload = function(e) {
				$('#upload-image-display').css('background-image',
					'url(' + e.target.result + ')');
				$('#upload-image-display').show();	

				$('#selected-image').attr('src', e.target.result);
				$('#selected-image').show();	
			}

			reader.readAsDataURL(input.files[0]);

			var formData = new FormData();
			formData.append("image", input.files[0]);

			$.ajax({
				url : '/upload',
				type: 'POST',
				data: formData,
				processData: false,
				contentType: false,
				success: function(data) {
					pred = data['pred']
					$('#prediction').text(pred);
					console.log(data)
					$('#heatmap-image-display').css('background-image',
						'url(/image/' + data['hm'][0] + ')');
					$('#heatmap-image-display').show();	
					$('#process-image-display').css('background-image',
						'url(/image/' + data['im_p'] + ')');
					$('#process-image-display').show();	
					$('#both-image-display').css('background-image',
						'url(/image/' + data['hm_im'][0] + ')');
					$('#both-image-display').show();	
				},
				error: function(jqXHR, textStatus, errorThrown) {
					console.log(jqXHR);
					console.log(textStatus);
					console.log(errorThrown);
					var err = 'Request failed: ' + errorThrown + '. Please contact morningsidelabs@gmail.com.';
					alert(err);
				}
			});
		}
	});

});
