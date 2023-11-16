{{- define "llmDeployment.name" -}}
   printf "%s-%s" "lorax" .Values.llmDeployment.name
{{- end -}}