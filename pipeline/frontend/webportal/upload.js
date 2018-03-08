$(document).ready(() => {

	var area = $('.panzoomed')[0];
	panzoom(area, {smoothScroll: false});

	function setSideClicks() {
		$('.side').click(function() {
			var bgi = $(this).css('background-image');
			bgi = bgi.replace('url(','').replace(')','').replace(/\"/gi, "");
			console.log(bgi)
			$('#selected-image').attr('src', bgi);
			//panzoom(area, {smoothScroll: false}).zoomAbs(0, 0, 1);
		});
	}
	

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

					console.log(data)

					for (var index in data['pred']) {
						var prob = data['pred'][index].toFixed(3);
						$('.predictions').append(
							'<p> ' + index + ' : ' + prob + '</p>');
					}

					$('#process-image-display').css('background-image',
						'url(/image/' + data['im_p'] + ')');
					$('#process-image-display').show();	

					for (var index in data['hm_im']) {
						$('.side-container').append(
							'<div class="side hm_im' + index + '"></div>');
						$('.hm_im' + index).css('background-image',
							'url(/image/' + data['hm_im'][index] + ')');
					}	

					setSideClicks();
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
