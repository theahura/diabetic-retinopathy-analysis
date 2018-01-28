$(document).ready(() => {
	
	function getImage(fp, callback) {
		$.ajax({
			url : '/image/' + fp,
			type: 'GET',
			processData: false,
			contentType: false,
			success: function(image) {
				callback(image);
			}
		});
	}

	$("#file-upload").change(() => {
		var input = $('#file-upload')[0];
		if (input.files && input.files[0]) {
			var reader = new FileReader();

			reader.onload = function(e) {
			  	$('#upload-image-display').attr('src', e.target.result);
				$('#upload-image-display').show();	
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
					getImage(data['hm'], (hm) => {
						$('#heatmap-image-display').attr('src', e.target.result);
						$('#heatmap-image-display').show();	
					});
					getImage(data['im_p'], (im) => {
						$('#process-image-display').attr('src', e.target.result);
						$('#process-image-display').show();	
					});
				}
			});
		}
	});

});
