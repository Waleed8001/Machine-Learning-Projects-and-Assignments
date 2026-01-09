package com.example.yoloproject

// Android permissions ke liye
import android.Manifest
import android.content.pm.PackageManager

// Graphics (Bitmap, Canvas, Paint, RectF etc.)
import android.graphics.*
import android.os.Bundle
import android.util.Log
import android.widget.Button
import android.widget.ImageView

// AppCompat activity
import androidx.appcompat.app.AppCompatActivity

// CameraX classes
import androidx.camera.core.*
import androidx.camera.lifecycle.ProcessCameraProvider

// Permission helpers
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat

// JSON parsing
import org.json.JSONArray

// IO & Networking
import java.io.ByteArrayOutputStream
import java.net.Socket
import java.nio.ByteBuffer

// Background thread executor
import java.util.concurrent.Executors

// ---------------- DATA CLASS ----------------
// YOLO se aane wali detection info ko hold karne ke liye
data class Detection(
    val className: String,   // Object ka naam (e.g. person, car)
    val confidence: Float,   // Confidence score
    val box: RectF           // Bounding box (left, top, right, bottom)
)

class MainActivity : AppCompatActivity() {

    // Python server ke sath TCP socket
    private var socket: Socket? = null

    // Camera frame processing ke liye background thread
    private val executor = Executors.newSingleThreadExecutor()

    // Laptop ka IP aur port jahan Python YOLO server chal raha hai
    private val HOST = "10.76.103.28"
    private val PORT = 9999

    // Camera output display karne ke liye ImageView
    private lateinit var imageView: ImageView

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        // UI layout set karna
        setContentView(R.layout.activity_main)

        // ImageView ko layout se link karna
        imageView = findViewById(R.id.imageView)

        // Start button click listener
        findViewById<Button>(R.id.startBtn).setOnClickListener {

            // Camera permission check
            if (ContextCompat.checkSelfPermission(
                    this,
                    Manifest.permission.CAMERA
                ) == PackageManager.PERMISSION_GRANTED
            ) {
                // Permission hai → camera + socket start
                startCamera()
                connectSocket()
            } else {
                // Permission nahi hai → request
                ActivityCompat.requestPermissions(
                    this,
                    arrayOf(Manifest.permission.CAMERA),
                    1
                )
            }
        }
    }

    // Permission ka result handle karna
    override fun onRequestPermissionsResult(
        requestCode: Int,
        permissions: Array<String>,
        grantResults: IntArray
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)

        // Agar camera permission mil gayi
        if (requestCode == 1 &&
            grantResults.isNotEmpty() &&
            grantResults[0] == PackageManager.PERMISSION_GRANTED
        ) {
            startCamera()
            connectSocket()
        }
    }

    // ---------------- SOCKET CONNECTION ----------------
    private fun connectSocket() {
        Thread {
            try {
                // Python server ke sath TCP connection
                socket = Socket(HOST, PORT)
                Log.d("SOCKET", "Connected to Python Server")
            } catch (e: Exception) {
                Log.e("SOCKET", "Connection failed", e)
            }
        }.start()
    }

    // ---------------- CAMERA START ----------------
    private fun startCamera() {

        // Camera provider lena
        val providerFuture = ProcessCameraProvider.getInstance(this)

        providerFuture.addListener({

            val provider = providerFuture.get()

            // Back camera select karna
            val cameraSelector = CameraSelector.DEFAULT_BACK_CAMERA

            // Image analysis use case
            val analysis = ImageAnalysis.Builder()
                // Latest frame hi process karo (old drop)
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .build()

            // Analyzer → har frame yahan aata hai
            analysis.setAnalyzer(executor) { image ->

                // Camera frame ko bitmap mein convert karna
                val sourceBitmap = image.toBitmap()

                if (sourceBitmap != null) {
                    try {
                        // Rotation fix karna (portrait / landscape)
                        val matrix = Matrix()
                        matrix.postRotate(
                            image.imageInfo.rotationDegrees.toFloat()
                        )

                        val rotatedBitmap = Bitmap.createBitmap(
                            sourceBitmap,
                            0,
                            0,
                            sourceBitmap.width,
                            sourceBitmap.height,
                            matrix,
                            true
                        )

                        // Default bitmap jo display hogi
                        var finalBitmapToShow = rotatedBitmap

                        socket?.let {
                            if (!it.isClosed) {

                                // Output stream → frame send karne ke liye
                                val output = it.getOutputStream()

                                // Input stream → detections receive karne ke liye
                                val input = it.getInputStream()

                                // ---------- STEP 1: IMAGE SEND ----------

                                // Bitmap ko JPEG mein compress karna
                                val jpegStream = ByteArrayOutputStream()
                                rotatedBitmap.compress(
                                    Bitmap.CompressFormat.JPEG,
                                    95,
                                    jpegStream
                                )

                                val jpegData = jpegStream.toByteArray()

                                // Pehle image ka size bhejna (4 bytes)
                                val sizeBytes = jpegData.size.toByteArray()
                                output.write(sizeBytes)

                                // Phir actual image bytes bhejna
                                output.write(jpegData)
                                output.flush()

                                // ---------- STEP 2: JSON RECEIVE ----------

                                // JSON size receive karna
                                val jsonSizeBytes = ByteArray(4)
                                val bytesRead = input.read(jsonSizeBytes)

                                if (bytesRead == 4) {
                                    val jsonSize =
                                        ByteBuffer.wrap(jsonSizeBytes).int

                                    val jsonBytes = ByteArray(jsonSize)
                                    input.read(jsonBytes)

                                    // JSON string banana
                                    val jsonString =
                                        String(jsonBytes, Charsets.UTF_8)

                                    // JSON → Detection objects
                                    val detections =
                                        parseDetections(jsonString)

                                    // ---------- STEP 3: DRAW BOXES ----------
                                    finalBitmapToShow =
                                        drawDetections(
                                            rotatedBitmap,
                                            detections
                                        )
                                }
                            }
                        }

                        // ---------- STEP 4: DISPLAY ----------
                        runOnUiThread {
                            imageView.setImageBitmap(finalBitmapToShow)
                        }

                    } catch (e: Exception) {
                        Log.e("SOCKET", "Send/Receive Error", e)
                    }
                }

                // Frame release karna (VERY IMPORTANT)
                image.close()
            }

            // Camera bind karna lifecycle ke sath
            try {
                provider.unbindAll()
                provider.bindToLifecycle(
                    this,
                    cameraSelector,
                    analysis
                )
            } catch (exc: Exception) {
                Log.e("CAMERA", "Binding failed", exc)
            }

        }, ContextCompat.getMainExecutor(this))
    }

    // ---------------- JSON PARSING ----------------
    private fun parseDetections(jsonString: String): List<Detection> {

        val detections = mutableListOf<Detection>()

        try {
            val jsonArray = JSONArray(jsonString)

            for (i in 0 until jsonArray.length()) {

                val obj = jsonArray.getJSONObject(i)

                val className = obj.getString("class_name")
                val confidence =
                    obj.getDouble("confidence").toFloat()

                val boxArray = obj.getJSONArray("box")

                val box = RectF(
                    boxArray.getDouble(0).toFloat(),
                    boxArray.getDouble(1).toFloat(),
                    boxArray.getDouble(2).toFloat(),
                    boxArray.getDouble(3).toFloat()
                )

                detections.add(
                    Detection(className, confidence, box)
                )
            }
        } catch (e: Exception) {
            Log.e("JSON", "Parsing error", e)
        }

        return detections
    }

    // ---------------- DRAW BOUNDING BOXES ----------------
    private fun drawDetections(
        bitmap: Bitmap,
        detections: List<Detection>
    ): Bitmap {

        val annotatedBitmap =
            bitmap.copy(Bitmap.Config.ARGB_8888, true)

        val canvas = Canvas(annotatedBitmap)

        detections.forEach { detection ->

            // Bounding box paint
            val boxPaint = Paint().apply {
                style = Paint.Style.STROKE
                strokeWidth = 8f
                color = Color.RED
            }

            // Box draw karna
            canvas.drawRect(detection.box, boxPaint)

            // Label text
            val label =
                "${detection.className} ${"%.2f".format(detection.confidence)}"

            val textPaint = Paint().apply {
                color = Color.WHITE
                textSize = 45f
            }

            val bgPaint = Paint().apply {
                color = Color.RED
            }

            val bounds = Rect()
            textPaint.getTextBounds(
                label, 0, label.length, bounds
            )

            // Label background
            canvas.drawRect(
                detection.box.left,
                detection.box.top - bounds.height() - 20,
                detection.box.left + bounds.width() + 20,
                detection.box.top,
                bgPaint
            )

            // Label text draw
            canvas.drawText(
                label,
                detection.box.left + 10,
                detection.box.top - 10,
                textPaint
            )
        }

        return annotatedBitmap
    }

    // ---------------- IMAGE → BITMAP ----------------
    private fun ImageProxy.toBitmap(): Bitmap? {

        // YUV format check
        if (format != ImageFormat.YUV_420_888) return null

        val yBuffer = planes[0].buffer
        val uBuffer = planes[1].buffer
        val vBuffer = planes[2].buffer

        val nv21 =
            ByteArray(
                yBuffer.remaining() +
                        uBuffer.remaining() +
                        vBuffer.remaining()
            )

        // NV21 format banana
        yBuffer.get(nv21, 0, yBuffer.remaining())
        vBuffer.get(nv21, yBuffer.remaining(), vBuffer.remaining())
        uBuffer.get(
            nv21,
            yBuffer.remaining() + vBuffer.remaining(),
            uBuffer.remaining()
        )

        val yuvImage = YuvImage(
            nv21,
            ImageFormat.YV12,
            width,
            height,
            null
        )

        val out = ByteArrayOutputStream()
        yuvImage.compressToJpeg(
            Rect(0, 0, width, height),
            95,
            out
        )

        val bytes = out.toByteArray()
        return BitmapFactory.decodeByteArray(bytes, 0, bytes.size)
    }

    // Int → 4 byte array (size send karne ke liye)
    private fun Int.toByteArray(): ByteArray =
        byteArrayOf(
            (this shr 24).toByte(),
            (this shr 16).toByte(),
            (this shr 8).toByte(),
            this.toByte()
        )

    // App destroy hone par cleanup
    override fun onDestroy() {
        super.onDestroy()
        executor.shutdown()
        socket?.close()
    }
}
