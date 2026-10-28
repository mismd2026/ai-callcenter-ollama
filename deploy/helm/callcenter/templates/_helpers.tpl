{{- define "cc.name" -}}{{ .Release.Name }}-callcenter{{- end -}}
{{- define "cc.labels" -}}
app.kubernetes.io/part-of: callcenter
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end -}}
