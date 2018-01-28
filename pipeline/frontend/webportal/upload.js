$(document).ready(() => {
	
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
					$('#heatmap-image-display').attr('src', '/image/' + data['hm']);
					$('#heatmap-image-display').show();	
					$('#process-image-display').attr('src', '/image/' + data['im_p']);
					$('#process-image-display').show();	
					$('#both-image-display').attr('src', '/image/' + data['hm_im']);
					$('#both-image-display').show();	
				}
			});
		}
	});

});
