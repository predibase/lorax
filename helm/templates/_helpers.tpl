{{- define "app.name" -}}
{{- printf "%s-%s" .Chart.Name .Release.Name | lower -}}
{{- end -}}