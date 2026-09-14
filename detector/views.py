"""
Views and REST API endpoints for Face Expression Recognition.
"""

import json
import csv
from django.shortcuts import render
from django.http import JsonResponse, HttpResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from django.core.files.base import ContentFile
from django.db.models import Count, Avg
from django.utils import timezone
import base64
import uuid

from .models import EmotionDetectionRecord
from .face_engine import engine, CLASS_NAMES, EMOTION_PALETTE


def index_view(request):
    """Renders the main single-page facial expression studio."""
    models_list = engine.get_available_models()
    recent_records = EmotionDetectionRecord.objects.all()[:12]
    total_scans = EmotionDetectionRecord.objects.count()
    
    # Calculate emotion distribution stats
    emotion_counts = {cls: 0 for cls in CLASS_NAMES}
    for row in EmotionDetectionRecord.objects.values('primary_emotion').annotate(count=Count('id')):
        if row['primary_emotion'] in emotion_counts:
            emotion_counts[row['primary_emotion']] = row['count']

    context = {
        'models': models_list,
        'class_names': CLASS_NAMES,
        'emotion_palette': EMOTION_PALETTE,
        'total_scans': total_scans,
        'emotion_counts': json.dumps(emotion_counts),
        'recent_records': [r.to_dict() for r in recent_records],
    }
    return render(request, 'detector/index.html', context)


@csrf_exempt
@require_http_methods(["POST"])
def api_predict(request):
    """
    Predicts emotion from image base64 payload or multipart file.
    Optionally saves the detection to history database.
    """
    try:
        model_name = request.POST.get('model_name', 'face_emotion_model.onnx')
        source = request.POST.get('source', 'webcam_live')
        save_record = request.POST.get('save_record', 'false').lower() == 'true'
        draw_annotations = request.POST.get('draw_annotations', 'true').lower() == 'true'

        image_data = None
        if 'image' in request.FILES:
            image_data = request.FILES['image']
        elif 'image_base64' in request.POST:
            image_data = request.POST['image_base64']
        else:
            # Check JSON body
            try:
                body = json.loads(request.body)
                image_data = body.get('image_base64')
                model_name = body.get('model_name', model_name)
                source = body.get('source', source)
                save_record = bool(body.get('save_record', save_record))
                draw_annotations = bool(body.get('draw_annotations', draw_annotations))
            except Exception:
                pass

        if not image_data:
            return JsonResponse({'success': False, 'error': 'No image data provided'}, status=400)

        # Run face engine
        result = engine.process_frame(
            image_data=image_data,
            model_name=model_name,
            draw_annotations=draw_annotations
        )

        record_id = None
        # Save record in database if requested or for snapshots/uploads
        if save_record and result['faces']:
            primary_face = result['faces'][0]
            record = EmotionDetectionRecord(
                source=source,
                model_name=model_name,
                primary_emotion=result['primary_emotion'],
                confidence=result['confidence'] / 100.0,
                probabilities={k: v / 100.0 for k, v in result['probabilities'].items()},
                bounding_box=primary_face.get('box', []),
                inference_time_ms=result['inference_time_ms'],
                face_count=result['face_count']
            )

            # Save face crop thumbnail if present
            crop_b64 = primary_face.get('face_crop_base64')
            if crop_b64 and ',' in crop_b64:
                header, data = crop_b64.split(',', 1)
                file_name = f"face_{uuid.uuid4().hex[:8]}.jpg"
                record.face_crop.save(file_name, ContentFile(base64.b64decode(data)), save=False)

            record.save()
            record_id = record.id
            result['record_id'] = record_id

        return JsonResponse(result)

    except Exception as e:
        return JsonResponse({'success': False, 'error': str(e)}, status=500)


@require_http_methods(["GET"])
def api_history(request):
    """Returns paginated detection history and aggregate stats."""
    emotion_filter = request.GET.get('emotion')
    source_filter = request.GET.get('source')
    limit = min(int(request.GET.get('limit', 20)), 100)
    offset = int(request.GET.get('offset', 0))

    queryset = EmotionDetectionRecord.objects.all()
    if emotion_filter and emotion_filter != 'all':
        queryset = queryset.filter(primary_emotion=emotion_filter)
    if source_filter and source_filter != 'all':
        queryset = queryset.filter(source=source_filter)

    total_count = queryset.count()
    records = queryset[offset:offset + limit]

    # Aggregate emotion counts
    agg_counts = {cls: 0 for cls in CLASS_NAMES}
    for row in EmotionDetectionRecord.objects.values('primary_emotion').annotate(c=Count('id')):
        if row['primary_emotion'] in agg_counts:
            agg_counts[row['primary_emotion']] = row['c']

    avg_conf = EmotionDetectionRecord.objects.aggregate(Avg('confidence'))['confidence__avg'] or 0.0

    return JsonResponse({
        'success': True,
        'total': total_count,
        'limit': limit,
        'offset': offset,
        'records': [r.to_dict() for r in records],
        'analytics': {
            'emotion_counts': agg_counts,
            'total_scans': EmotionDetectionRecord.objects.count(),
            'average_confidence': round(avg_conf * 100, 1)
        }
    })


@csrf_exempt
@require_http_methods(["POST", "DELETE"])
def api_history_delete(request, record_id=None):
    """Deletes a specific history record or all history."""
    try:
        if record_id:
            EmotionDetectionRecord.objects.filter(id=record_id).delete()
            return JsonResponse({'success': True, 'message': f'Record {record_id} deleted.'})
        else:
            # Clear all
            EmotionDetectionRecord.objects.all().delete()
            return JsonResponse({'success': True, 'message': 'All detection records cleared.'})
    except Exception as e:
        return JsonResponse({'success': False, 'error': str(e)}, status=500)


@require_http_methods(["GET"])
def api_models_info(request):
    """Returns list of models and available metadata."""
    models = engine.get_available_models()
    return JsonResponse({
        'success': True,
        'models': models,
        'classes': CLASS_NAMES,
        'palette': EMOTION_PALETTE
    })


@require_http_methods(["GET"])
def api_export_history(request):
    """Exports detection history as CSV or JSON."""
    export_format = request.GET.get('format', 'csv').lower()
    records = EmotionDetectionRecord.objects.all()

    if export_format == 'json':
        data = [r.to_dict() for r in records]
        response = JsonResponse(data, safe=False, json_dumps_params={'indent': 2})
        response['Content-Disposition'] = f'attachment; filename="emotion_detections_{timezone.now().strftime("%Y%m%d_%H%M%S")}.json"'
        return response

    # Default CSV
    response = HttpResponse(content_type='text/csv')
    response['Content-Disposition'] = f'attachment; filename="emotion_detections_{timezone.now().strftime("%Y%m%d_%H%M%S")}.csv"'

    writer = csv.writer(response)
    writer.writerow(['ID', 'Timestamp', 'Source', 'Model', 'Primary Emotion', 'Confidence (%)',
                     'Angry (%)', 'Fear (%)', 'Happy (%)', 'Sad (%)', 'Surprise (%)', 'Inference ms'])

    for r in records:
        probs = r.probabilities or {}
        writer.writerow([
            r.id,
            r.created_at.strftime('%Y-%m-%d %H:%M:%S'),
            r.source,
            r.model_name,
            r.primary_emotion,
            round(r.confidence * 100, 1),
            round(probs.get('Angry', 0) * 100, 1),
            round(probs.get('Fear', 0) * 100, 1),
            round(probs.get('Happy', 0) * 100, 1),
            round(probs.get('Sad', 0) * 100, 1),
            round(probs.get('Suprise', 0) * 100, 1),
            round(r.inference_time_ms, 1)
        ])

    return response
